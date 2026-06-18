# Integrating IR-level GNN Context with oss-fuzz-gen

This document covers how the PDG slice summarizer connects to automated harness
generation (specifically [oss-fuzz-gen](https://github.com/google/oss-fuzz-gen)),
what problem it solves for small sparse language models, and what the integration
looks like in practice.

**Note on scope.** This is an *application tool*, not a training experiment. The
PDG slice summarizer (`slice_context.py`) extracts structured vulnerability context
from any IR function and formats it for LLM prompt injection. It shares the
`preprocess_slice_pdg_v3.py` preprocessor with the §23 GNN architecture experiment
(sink-node readout + CD depth cap), but the two are distinct: §23 is a Devign
classifier variant; the context enrichment is downstream tooling for harness
generation that works regardless of which GNN checkpoint is used for scoring.

---

## The core problem: small models need pre-computed context

SCAR uses Qwen3 35B-A3B — a 35B parameter model with 3B *active* parameters due to
mixture-of-experts routing. At inference time this is a 3B model: capable, but not
capable of reasoning from scratch about data flow through binary IR. If you hand it
an LLVM IR function and ask "write a fuzzing harness that finds the vulnerability",
the model has to:

1. Parse the IR structure
2. Trace data flow from input to dangerous operations
3. Identify whether a guard check is present
4. Determine what values would bypass or overwhelm the check
5. Write valid libFuzzer harness code

Steps 1–4 are the hard part. Step 5 is the easy part. A 3B-active model can do
step 5 reliably — it's constrained generation from a specification. Steps 1–4
at 3B active parameters are unreliable because they require reasoning across the
full IR graph structure, which no language model processes natively.

**The PDG slice summarizer pre-computes steps 1–4.** It converts the IR graph into
a structured vulnerability specification that the model can read as plain text and
use directly for code generation.

---

## What the summarizer produces

Given any LLVM IR function, `slice_context.py` extracts the PDG backward slice
from dangerous sinks, then produces:

```
============================================================
GNN Vulnerability Context
Suspicion score : 87.3%  (SUSPICIOUS)
Sinks           : free — frees heap memory — double-free or use-after-free if ca
Input channels  : external_call_return
Guard status    : NO icmp in slice — sink appears UNGUARDED
Harness target  : fuzz call target twice with same pointer; access after free
Slice           : 13 nodes, 3 sink(s)
============================================================
```

Real output on `ir/doublefree_vuln.ll` (double-free of a `malloc`'d pointer):

```
============================================================
GNN Vulnerability Context
Sinks           : malloc — allocates heap — return may be null; size may overflow;
                  free — frees heap memory — double-free or use-after-free if ca;
                  free — frees heap memory — double-free or use-after-free if ca
Input channels  : external_call_return
Guard status    : NO icmp in slice — sink appears UNGUARDED
Harness target  : fuzz size=0; size=SIZE_MAX; integer overflow in size computation
                | fuzz call target twice with same pointer; access after free
                | fuzz call target twice with same pointer; access after free
Slice           : 13 nodes, 3 sink(s)
============================================================
Natural language:
  Function `main` contains: `malloc` (allocates heap — return may be null; size
  may overflow); `free` (frees heap memory — double-free or use-after-free if
  called again); `free` (frees heap memory — double-free or use-after-free if
  called again). Input originates from: external_call_return. Guard status: no
  comparison (icmp) in slice — sink appears UNGUARDED. Slice: 13 nodes, 3
  dangerous sink(s).
```

Real output on `ir/oob_read_vuln.ll` (GEP with non-constant index, no bounds check):

```
============================================================
GNN Vulnerability Context
Sinks           : getelementptr — pointer arithmetic with non-constant index — out-of-bou;
                  printf — format string to stdout — dangerous if format is user-c
Input channels  : external_call_return
Guard status    : NO icmp in slice — sink appears UNGUARDED
Harness target  : fuzz index at, near, and beyond array bounds; negative index; index=SIZE_MAX
                | fuzz user-controlled format string with %%n, %%s, %%x
Slice           : 11 nodes, 2 sink(s)
============================================================
```

The `natural_language` field is the harness-generation prompt injection block. The
`harness_hint` field is the concrete fuzzing guidance for the model.

---

## Integration points

### 1. oss-fuzz-gen function prioritisation

oss-fuzz-gen's [Fuzz Introspector](https://github.com/ossf/fuzz-introspector) scores
functions by **coverage reachability** — how many new code paths does a harness for
this function unlock? This is a coverage oracle.

The GNN provides a complementary **vulnerability oracle**: how likely is this function
to contain a memory safety bug? These are orthogonal signals. A function can be:

- High coverage reach, low suspicion — good harness target for coverage; unlikely to
  find bugs
- Low coverage reach, high suspicion — the bug is here but hard to trigger
- High reach, high suspicion — highest priority: both likely to find bugs and to
  exercise new paths

Combined scoring:

```
priority(fn) = α · coverage_score(fn) + (1 - α) · gnn_score(fn)
```

`gnn_score` is the sigmoid output of `model_slice_pdg.pt`. `α ≈ 0.5` gives equal
weight; tune toward the vulnerability oracle when the objective is bug-finding rather
than coverage measurement.

### 2. Structured context for harness generation LLM

The oss-fuzz-gen harness generation prompt currently includes:

- Function signature
- Return type and parameter types
- Brief natural-language description (if available)
- Example usage (if available)

What it does *not* include: which specific operation in the function is dangerous,
whether input reaches that operation without a guard, and what values trigger the bug.

The PDG slice summarizer adds exactly that. Injection point: between the function
signature and the harness generation instruction.

**Before (oss-fuzz-gen default):**

```
Generate a libFuzzer harness for the following C function:

  void process_packet(char *buf, int len) { ... }

The harness should:
- Call the function with fuzz-controlled inputs
- Link against the target library
```

**After (with GNN context):**

```
Generate a libFuzzer harness for the following C function:

  void process_packet(char *buf, int len) { ... }

[GNN Vulnerability Context]
Suspicion score : 91.4%  (SUSPICIOUS)
Sinks           : memcpy — copies n bytes from src to dest — no overlap or bounds check
Input channels  : function_argument
Guard status    : NO icmp in slice — sink appears UNGUARDED
Harness target  : fuzz n relative to dest buffer size; n=0, n=SIZE_MAX, n=dest_size+1

The harness should:
- Call the function with fuzz-controlled inputs
- Prioritise testing the memcpy length argument: try values at, near, and beyond
  the destination buffer boundary
- Link against the target library
```

The model now does constrained generation from a specification rather than open-ended
data flow reasoning. This is the step-5 problem, not the steps-1–4 problem.

---

## CLI usage

**Standalone context extraction (no trained model needed):**

```bash
python3 train_gnn/slice_context.py path/to/function.ll
python3 train_gnn/slice_context.py path/to/function.ll --json   # structured output
```

**Scored output with context block (`model_slice_pdg.pt` required):**

```bash
python3 train_gnn/scan_ir.py function.ll --context
python3 train_gnn/scan_ir.py file.ll --all-functions --context
```

Example scan output with `--context`:

```
process_packet.ll  [31 nodes]  91.4%  →  VULNERABLE

============================================================
GNN Vulnerability Context
Suspicion score : 91.4%  (SUSPICIOUS)
Sinks           : memcpy — copies n bytes from src to dest — no overlap or bounds check
Input channels  : function_argument
Guard status    : NO icmp in slice — sink appears UNGUARDED
Harness target  : fuzz n relative to dest buffer size; n=0, n=SIZE_MAX, n=dest_size+1
Slice           : 31 nodes, 1 sink(s)
============================================================
```

**As a library:**

```python
from preprocess_slice_pdg import ir_to_graph_slice_pdg
from slice_context import summarize_slice, format_for_llm

ir_text = open("function.ll").read()
g = ir_to_graph_slice_pdg(ir_text)
if g is not None:
    summary = summarize_slice(g, fn_name="process_packet")
    context_block = format_for_llm(summary, score=0.914)
    # inject context_block into your LLM prompt
```

---

## What still needs to be built

The infrastructure above produces the vulnerability specification. The full
integration requires:

1. **oss-fuzz-gen fork or plugin** that calls `scan_ir.py --context` per candidate
   function and injects the output into the harness generation prompt. The oss-fuzz-gen
   LLM pipeline already has a prompt construction layer; this is an injection before
   the generation call.

2. **Combined scoring** in oss-fuzz-gen's function ranking: `fuzz_introspector_score +
   gnn_score`. Fuzz Introspector already exports per-function scores as JSON; the GNN
   score is one additional field.

3. **Node-level supervision** (long-term): current GNN produces a function-level
   suspicion score. With instruction-level labels from SCAR (planted bug IR + exact
   sink instruction location), the model could be retrained to produce per-node scores,
   identifying not just "this function is suspicious" but "this specific `memcpy` call
   on line N is the unguarded sink." The sink-node readout architecture (§23 experiment)
   and the `sink_mask` field in `preprocess_slice_pdg_v3.py` are already structured
   for this upgrade — they only require instruction-level training labels to realise it.
   The context enrichment tooling would also benefit directly: per-node scores would
   replace the current static sink-identification heuristic in `slice_context.py`.

---

## Why this is a real contribution

Fuzz Introspector solves: *which functions are hardest to cover with existing harnesses?*

The GNN + PDG slice solves: *which functions contain dangerous patterns, and specifically
what are those patterns?*

Neither exists in isolation in the oss-fuzz-gen pipeline today. The combination —
coverage oracle for harness prioritisation, vulnerability oracle for harness targeting —
is what transforms automated harness generation from coverage-maximisation to
bug-finding.

The pre-computed context block is the specific contribution: it converts the hard
IR-level analysis (data flow, sink identification, guard detection) into a text
specification that a small sparse model can act on directly. Without it, you are
asking a 3B-active model to reason about IR structure; with it, you are asking it
to generate code from a spec.
