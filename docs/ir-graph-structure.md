# Instruction-Level IR Graph Structure

Visual reference for the graph representation built by `preprocess_instr_v3.py` (§14).
Each C function is compiled to LLVM IR and converted to a multi-relational graph
with four edge types passed to an RGCN classifier.

---

## Example: `char read_and_write(char *buf, int idx)`

```c
char read_and_write(char *buf, int idx) {
    char old = buf[idx];      // load
    buf[idx] = old + 1;       // store — same pointer as load
    return old;
}
```

```mermaid
flowchart TD
    VCN(["⬡  0 · Virtual Context Node"])

    subgraph args [" Function Arguments "]
        A1["▷  %buf · char*"]
        A2["▷  %idx · int"]
    end

    subgraph consts [" Constants "]
        C1["◇  i8 · 1"]
    end

    subgraph entry [" Instructions — entry block "]
        I1["GEP · getelementptr i8, char* %buf, int %idx\n→ %ptr"]
        I2["LOAD · load i8, i8* %ptr\n→ %old"]
        I3["ADD · add i8 %old, 1\n→ %inc"]
        I4["STORE · store i8 %inc, i8* %ptr"]
        I5["RET · ret i8 %old"]
    end

    %% CFG — sequential control flow within block
    I1 -->|" CFG "| I2
    I2 -->|" CFG "| I3
    I3 -->|" CFG "| I4
    I4 -->|" CFG "| I5

    %% DFG — SSA def-use: edge from definition to each use
    A1 -.->|" DFG "| I1
    A2 -.->|" DFG "| I1
    I1 -.->|" DFG "| I2
    I1 -.->|" DFG "| I4
    I2 -.->|" DFG "| I3
    C1 -.->|" DFG "| I3
    I3 -.->|" DFG "| I4
    I2 -.->|" DFG "| I5

    %% State — memory ordering between load/store on same pointer (§14)
    I2 ==>|" State "| I4

    %% Global — VCN bidirectional to every node
    VCN <-->|" Global "| I1
    VCN <-->|" Global "| I2
    VCN <-->|" Global "| I3
    VCN <-->|" Global "| I4
    VCN <-->|" Global "| I5
    VCN <-->|" Global "| A1
    VCN <-->|" Global "| A2
    VCN <-->|" Global "| C1

    classDef vcn    fill:#e2e8f0,stroke:#475569,stroke-width:2px,font-weight:bold
    classDef arg    fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px
    classDef const  fill:#fef9c3,stroke:#ca8a04,stroke-width:1.5px
    classDef instr  fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px
    classDef memop  fill:#fce7f3,stroke:#db2777,stroke-width:2px

    class VCN vcn
    class A1,A2 arg
    class C1 const
    class I1,I3,I5 instr
    class I2,I4 memop
```

---

## Node Types

| Color | Type | Description | Vocab ID |
|---|---|---|---|
| Grey | Virtual Context Node | One per graph; global hub reducing diameter to O(1) | 0 |
| Blue | Function Argument | One node per function parameter | 1 |
| Yellow | Constant (int/fp) | Literal values; carry Perfograph magnitude encoding | 76 / 77 |
| Green | Instruction | Most IR instructions (GEP, ADD, RET, icmp, br, …) | 2–74, 80–105 |
| Pink | Memory instruction | `load` and `store` — highlighted because they participate in State edges | 27 / 28 |
| — | Call target (Alloc) | malloc, calloc, realloc, … | 106 |
| — | Call target (Copy) | memcpy, memmove, memset | 107 |
| — | Call target (String) | strcpy, sprintf, gets, … | 108 |
| — | Call target (FileIO) | fopen, read, write, … | 109 |
| — | Call target (Network) | recv, send, accept, … | 110 |

---

## Edge Types

| Type | Arrow | Relation ID | What it encodes |
|---|---|---|---|
| CFG | solid `→` | 0 | Sequential instruction execution; inter-block branch targets |
| DFG | dashed `⤳` | 1 | SSA def-use: each instruction's result flows to instructions that consume it |
| Global | double `↔` | 2 | Virtual Context Node to every other node (bidirectional); collapses graph diameter to O(1) so a local patch propagates in two message-passing steps |
| State | thick `⟹` | 3 | Memory operation ordering: consecutive `load`/`store` pairs on the same pointer, in execution order |

---

## Why the State edge matters

DFG already connects `GEP → LOAD` and `GEP → STORE` because both use `%ptr` as an
operand. What DFG does *not* encode is their ordering relative to each other. The State
edge `LOAD ⟹ STORE` is the first edge type that says: **this memory location was read,
then written**.

In a use-after-free scenario the pattern is reversed — the pointer is freed between the
load and the store, but the graph topology of the load/store pair is identical to the safe
case. State edges give the RGCN a distinct relation to learn from, rather than routing
all memory semantics through the same DFG weight matrix.

---

## Mapping to the model

```
preprocess_instr_v3.py          train_instr_v3.py
────────────────────────        ─────────────────────────────────────
Pass 1 — allocate nodes    →    nn.Embedding(111, 128)  # opcode ID
Pass 2 — CFG edges (0)     →    RGCNConv(129, 64, num_relations=4)
Pass 3 — DFG edges (1)     →    RGCNConv( 64, 64, num_relations=4)
         + Perfograph mag   →    x[:,1] appended after embedding
Pass 4 — Global edges (2)  →    relation weight matrix W_2
Pass 5 — State edges (3)   →    relation weight matrix W_3  ← §14
```

Full experiment log: [`docs/ir-embed.md`](ir-embed.md)
