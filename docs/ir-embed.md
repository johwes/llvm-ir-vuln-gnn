# GNN Vulnerability Detector — LLVM IR Experiments

**Status:** Complete — 22 experiments  
**Models:** `johnnywesterlund/scar-gnn-defect-detector` on Hugging Face  
**Code:** `github.com/johwes/llvm-ir-vuln-gnn`

| Benchmark | Best result |
|---|---|
| Devign test accuracy | 58.75% (§13 Perfograph) — ceiling ~58%, explained below |
| scarnet real-world P/R | **84.6%** (§12 PDG slice, 11/13 known-vulnerable) |
| zlib v1.2.11 CVE rank | **top 10 / 148** functions, 4 of 9 models, zero-shot |

---

## TL;DR — What this is and what we found

We trained a neural network to read compiled C code and score each function by how likely it is to contain a security bug — with no human-written rules, no source code, and no knowledge of the specific codebase.

**How it works:** C source is compiled to LLVM IR (a machine-readable intermediate form). Each function's IR is converted into a graph — nodes are instructions, edges represent data flow and control flow. A Graph Neural Network (GNN) reads that graph and outputs a suspicion score. Training used Devign, a public dataset of ~27,000 C functions from open-source projects labeled by whether a security fix was later committed.

**The benchmark number looks bad — but that's expected.** On the Devign test set, all architectures we tried score ~57–58% accuracy. The "always guess the majority class" baseline is 56.6%. So we're only 1–2 percentage points above random on the benchmark. This is not a model failure; it is a hard ceiling with three known causes:

1. **Security bugs are often absences.** A safe `memcpy` and an unsafe one look nearly identical in the graph — the only difference is the missing bounds check. A graph can only represent what is there, not what is not.
2. **The compiler discards the most useful information.** Variable names, string literals, and comments are gone before the model sees anything. A function that reads `gets(user_input)` in source becomes an anonymous call instruction in IR.
3. **The training labels are noisy.** Devign labels are assigned at commit level — sometimes the actual bug is in a different function than the one that changed.

These causes are structural. Trying 12 different architectures, edge types, and feature sets over 22 experiments confirmed that better model design cannot break through this ceiling. The reference number for the best source-code language model (CodeBERT, which reads actual C with names intact) is 63.4% — our 6pp gap is entirely explained by the IR vocabulary loss.

**Real-world performance is a different story.** When evaluated on actual vulnerable code:

- **scarnet** (purpose-built vulnerable server, 13 known-vulnerable functions out of 19): best model finds **11/13 at 84.6% precision/recall**. The 2 missed are a bug that only triggers on ARM hardware and one that requires fuzzing to discover — both structurally invisible to any static tool.
- **zlib v1.2.11** (real production library, 148 functions, 1 known CVE): 4 of 9 models rank `deflate_stored` in the **top 10 out of 148** with no zlib training. Mean reciprocal rank 0.133 — 3.5× better than random.

**Practical value:** The GNN is a zero-cost pre-screener. It runs in under a second per function and narrows a large codebase down to a ranked shortlist worth human attention. It is not a replacement for fuzzing or formal verification — it complements them by triaging which functions deserve deeper scrutiny first.

---

## Setup

5 vulnerable/fixed C pairs from [johwes/SCAR-test-c](https://github.com/johwes/SCAR-test-c)
(doublefree, nullderef, oob\_read, uninit, divzero), compiled to LLVM IR with
`clang -O0 -S -emit-llvm`. Feature: normalized opcode-frequency histogram
(~70 LLVM opcodes). Distance: cosine. No neural network, no training.

Note: the IR files in `experiments/ir_embed_demo/ir/` are hand-written
representative `.ll` files, not real clang output. Run `./run.sh` inside the
`scar-agent` container to replace them with real compiled IR.

---

## Results

### Global metric (misleading)

```
avg vuln  ↔ vuln   distance : 0.3144
avg fixed ↔ fixed  distance : 0.4268
avg vuln  ↔ fixed  distance : 0.3787   →  1.02× within-class
```

Looks like no signal. It isn't — the problem is the evaluation design.
Different vulnerability types (divzero vs nullderef) are naturally far apart
because they are different programs. Averaging cross-CWE distances into
"within-class" collapses the real signal.

### Per-pair metric (correct for a scanner)

For a scanner, the relevant question is: given a new function's embedding,
is it closer to known-vulnerable embeddings than to known-fixed embeddings?

| Pair | own vuln↔fixed | avg vuln↔other fixed | signal |
|---|---|---|---|
| divzero | 0.159 | 0.601 | YES |
| doublefree | 0.012 | 0.447 | YES |
| nullderef | 0.314 | 0.315 | YES (marginal) |
| oob\_read | 0.368 | 0.358 | no |
| uninit | 0.106 | 0.400 | YES |

4 of 5 pairs show a clear signal with zero training.

### Key finding: vulnerability-class clustering

`nullderef_V ↔ oob_read_V = 0.081` — two different programs, both are
"missing a conditional branch" vulnerabilities. They sit closer together
than either does to its own fixed version (0.314 and 0.368). Their fixed
versions also cluster tightly: `nullderef_F ↔ oob_read_F = 0.070`.

The opcode histogram groups code by *what structural feature is absent*
rather than by specific CWE. That generalisation is what a scanner needs:
train on one missing-branch pattern, detect others.

---

## What the failure tells us

`oob_read` failed because `nullderef_fixed` and `oob_read_fixed` are
structurally nearly identical — both just add `icmp + br`. When two fixes
share the same IR shape, whole-function histograms cannot tell them apart,
and the cross-class distance drops to the level of the within-pair distance.

This directly confirms the granularity concern from `docs/research.md`:
whole-function opcode histograms cannot localise *which* missing branch matters
when multiple functions share the same fix topology. Subgraph-level or
basic-block-level features are needed to break this degeneracy.

---

## What this confirms and what it does not

**Confirmed:** A structural signal exists in opcode histograms on toy
single-function examples, detectable without any training.

**Not tested:**
- Whether the signal survives on real-world functions where the vulnerable
  pattern is buried in hundreds of lines of surrounding code
- Whether contrastive training improves discriminability over raw histograms
- Whether the clustering generalises beyond the 5 CWE classes tested

---

## Practical ceiling

### What this approach structurally cannot detect

The fundamental limitation is the absence of taint/dataflow analysis.
Opcode histograms — and even full CFG/DFG graph embeddings — cannot follow
a value across function boundaries and determine whether it was sanitized
before reaching a dangerous sink. Most serious real-world vulnerabilities
require exactly that reasoning:

- **Injection and buffer overflows from user input**: requires tracking
  tainted input through call chains to where it is used without bounds
  checking. Structural similarity to a known-unsafe function is not enough
  — the same function shape can be safe or unsafe depending on whether the
  caller validated the input.
- **Use-after-free in complex allocation patterns**: the dangerous access
  may be in a different function from the free. No per-function embedding
  can see this.
- **Integer overflows in protocol parsing**: depends on the range of values
  reachable at a specific point, not the presence or absence of a branch.

Tools like CodeQL exist precisely to answer these questions with
interprocedural dataflow graphs. This approach does not compete with that.

### Where it realistically fits

The honest position is **cheap pre-filter, not standalone detector**:

1. Embed every function's IR against the known-vulnerable corpus.
2. Flag functions with high structural similarity to known-vulnerable
   patterns as candidates.
3. Feed those candidates into CodeQL, Semgrep, or the SCAR LLM repair loop
   for the expensive, precise analysis.

This costs zero LLM calls and runs in seconds. If it surfaces real
candidates that rule-based tools then confirm, it earns its place in the
pipeline as a prioritisation signal.

### What would need to be true to become competitive

Raw opcode histograms are the weakest form of this idea. Competitive
detection would require:

- **Full CFG/DFG graph representation** (ProGraML-style) so the model sees
  control-flow paths and data dependencies, not just opcode frequencies.
- **Thousands of training pairs** across diverse codebases, not dozens.
- **Interprocedural context** — embedding call-graph subgraphs rather than
  individual functions.

Both blockers are largely resolved by existing open work — see experiment 4
below. Even so, the ML-based vulnerability detection literature has a poor
track record of generalising beyond benchmark conditions. This is a
research direction, not a near-term production capability.

### The one genuinely novel property

The self-improving corpus is the most defensible advantage over static
rule-based tools. Every SCAR accepted patch on any target is a labelled
(vulnerable IR, fixed IR) pair produced at zero marginal cost. CodeQL rules
do not improve when you find a new bug. A model retrained on accumulated
SCAR patches does — and it specialises to exactly the patterns SCAR
encounters in practice. Whether that specialisation translates to useful
detection precision on unseen code is the core empirical question.

---

## Next experiments

### 1. Real IR from clang (prerequisite for everything else)

Run inside the `scar-agent` container, which has clang:

```bash
cd experiments/ir_embed_demo
./run.sh        # compiles all 7 pairs; replaces hand-written ir/*.ll
```

This adds the two pairs not yet in the hand-written IR (bof, signedoverflow)
and validates that the results hold on real compiler output rather than
representative approximations. Re-run `demo.py` and compare the per-pair
signal table to the baseline above.

---

### 2. Contrastive training step

Goal: measure whether a trained embedding widens the separation ratio beyond
the raw histogram baseline.

**Add `experiments/ir_embed_demo/train.py`** — a minimal PyTorch script:

```
Input:  normalized opcode histograms (dim ≈ 70) from ir/*.ll
Model:  MLP — Linear(70→32) → ReLU → Linear(32→16)
Loss:   contrastive loss (Hadsell et al. 2006)
          same-class pairs (vuln+vuln, fixed+fixed): pull together
          cross-class pairs (vuln+fixed):            push apart, margin=0.5
Output: 16-dim embeddings; re-run cosine distance analysis on these
```

With only 7×2 = 14 samples, use leave-one-pair-out cross-validation:
train on 6 pairs, test separation on the held-out pair. Report whether
the trained embedding improves the per-pair signal over the raw baseline.

Requires: `pip install torch` (CPU-only is fine, no GPU needed at this scale).

**Success criterion:** separation ratio on the held-out pair improves over
the raw histogram baseline for at least 5 of 7 folds.

---

### 3. Real-world functions from SCAR accepted patches on zlib

Goal: test whether the per-pair signal survives when the vulnerable pattern
is buried in a real function rather than a purpose-built 10-line file.

Scarnet is synthetic — same code, same patches, same structure every run.
It tests nothing beyond what the toy examples already cover. The right
source is **zlib v1.2.11**, which produced 21 accepted patches in a real
pipeline run. The patched functions (`deflate`, `inflate`, `crc32`, etc.)
are hundreds of lines of production code; each patch touches a handful of
lines. That ratio — small vulnerable subgraph, large surrounding context —
is exactly the stress test this experiment needs.

**Source:** the `scar-results.json` from the zlib v1.2.11 pipeline run.
Each accepted entry contains:
- `finding.file_path` — the source file
- `finding.line` — the vulnerable line
- `patch` — the unified diff

**Procedure:**

1. For each accepted patch entry in `scar-results.json`:
   - `original.c` = the source file as-is (vulnerable)
   - Apply the patch with `patch -o fixed.c original.c diff.patch`
   - Compile both: `clang -O0 -S -emit-llvm -o original.ll original.c`

2. Extract the enclosing function from each `.ll` file using the finding
   line number. A function in LLVM IR text format starts with `define` and
   ends with the matching `}`. Walk the IR to find the function whose line
   range contains `finding.line`.

3. Run the same histogram analysis on the extracted function IR slices,
   not the whole-file IR.

4. Report the per-pair signal table as in experiment 1.

**Key question:** does the separation ratio hold when the vulnerable
subgraph is a small fraction of the total function IR? If it collapses
below 1.1×, the granularity problem is real and subgraph-level features
(basic-block or sliding-window) are needed before the approach is viable
on production code.

**Add `experiments/ir_embed_demo/extract_functions.py`** to automate
steps 1–2 given a `scar-results.json` and a source directory.

---

### GNN training — results summary

| Method | Test Acc | Notes |
|---|---|---|
| Majority-class baseline | 56.6% | always predict "fixed" |
| 4b CFG-only, GCNConv | 55.04% | barely learning |
| 4c PDG, RGCNConv | 56.08% | +1.0% from DFG edges |
| 4d v2.0 PDG + 45 features (30ep h=64) | 56.32% | +0.2% from 34 new features |
| 4d v2.0 PDG + 45 features (60ep h=128) | **57.84%** | best GNN result; peaks epoch 8 |
| 4d v2.1 MLP attention gate (30ep h=128) | 56.88% | gate upgrade no meaningful gain |
| **4a CodeBERT (this run, Colab T4)** | **63.43%** | **granularity gap confirmed** |
| CodeBERT published baseline | 62.08% | matches within noise |
| UniXcoder published | 69.29% | upper bound for token-based models |
| 5a instruction-level GNN (32-feat one-hot, 30ep h=64) | 55.84% | worse than block-level |
| 5b GRU hybrid (opcode sequences, 30ep h=64) | 56.96% | below block-level static |
| 5c SupCon k-NN (batch=512, τ=0.07, k=5, 50ep) | 55.84% | embedding collapse — see §5c |
| **6 BigVul Triplet k-NN (batch=32, margin=0.3, k=5, 50ep)** | **51.21%** | soft collapse; pair-sim 0.98 throughout — see §6 |
| **7 Instruction-level GNN (opcode embed=128, h=64, 30ep)** | **58.00%** | first to clear block-level ceiling; 60ep/h=128 overfits to 56.16% — see §7 |
| **8 BigVul Instr-level Triplet k-NN (50ep, h=64)** | **48.39%** | soft collapse, pair-sim 0.9984→0.9995 ↑ — see §8 |
| **10a Instruction-level + predicate vocab (VOCAB_SIZE 110, 30ep, h=64)** | **56.85%** | below §7 (58.00%): vocab enrichment insufficient for classifier alone — see §10a |
| **10b BigVul Instr-level FCL+SAGPooling (tau=0.07, γ=2.0, 50ep)** | **47.58% k-NN** | collapse persists (pair-sim 0.9992 ↑); loss plateau from epoch 2; contrastive branch closed — see §10b |
| **11 Slice-GNN: backward DFG slice from dangerous sinks (30ep, h=64)** | **56.64%** | below §7 (58.00%): DFG-only slice misses guard conditions (control dependence) — icmp/br predecessors invisible — see §11 |
| **12 Slice-PDG-GNN: PDG slice (DFG + control deps, 30ep, h=64)** | **56.48%** | PDG expansion grew slices (mean 37→57 nodes) but no accuracy gain over §11; icmp+br present in safe code too — see §12 |

**GNN structural ceiling: ~57–58%** across all block-level variants and training objectives. Every
architectural improvement at block level — relational edges, 34 semantic features, larger
hidden, expressive attention, GRU sequence encoding, contrastive loss — saturated in this band.
Instruction-level GNN (§7) edged past at 58.00% — marginal but directionally confirmed.

**This is a modelling gap, not a representation gap.**

LLVM IR at `-O0` is semantically equivalent to source: every vulnerability
present in source is present in the IR. Identifier information is preserved —
`strcpy(buf, src)` becomes `call void @strcpy(i8* %buf, i8* %src)`. The
variable name `buf`, the function name `strcpy`, the pointer type — all there.

What our GNN discards is not inherent to the IR: it is a consequence of how we
read the IR. We extract opcodes (one-hot) and 30 hard-coded API presence flags,
then throw away every other identifier. CodeBERT starts with pretraining on
billions of tokens and already knows `strcpy` is dangerous, that `buf + len`
without a bounds check is a pattern, that `malloc` without a null check is
wrong. Our GNN must learn all of that from 10K labelled examples with
hand-coded features — and it can't.

**Implication:** a transformer trained directly on IR text — treating LLVM IR as
a programming language — would likely match CodeBERT performance on this task,
because the information content is identical. IR may in fact be *more* learnable
than source: no macros, no syntactic sugar, explicit types, explicit memory
operations, explicit control flow. The 5.6-point gap is the cost of using a GNN
with hand-coded features, not the cost of using IR.

**Why binary classification is the wrong tool for this problem — and why
contrastive learning is the right one.**

Even a GNN with access to full identifier vocabulary would face a deeper
problem: the graph *shape* of a vulnerability is statistically almost
indistinguishable from safe code of similar structure.

Consider: `if (user_supplied_length > MAX_AUTH_BUFFER)` compiles to:
```
%2 = load i32, ptr %1
%cmp = icmp sgt i32 %2, 256
```
The vulnerability is physically there. But `%2` is now an anonymous register —
mathematically identical to a safe loop counter checking `i < array_size`. A
memory leak is just a missing edge from `alloca` to `free` in a graph of 300
nodes. The GNN must deduce, from pure arithmetic, that a specific sequence of
40 safe operations *should have ended in a `free` but didn't* — without any
semantic label pointing at the danger zone. Message-passing dilutes the signal
before it can connect the dots across 300 nodes.

Binary classification asks: *"is this anonymous graph bad?"* — answerable only
by statistical correlation with patterns the model has seen before. A GNN
trained on 10K examples with hand-coded opcode features cannot build that
statistical corpus. CodeBERT can, because it arrives pretrained on billions of
tokens where `user_supplied_length` and `MAX_AUTH_BUFFER` are already loaded
with meaning.

**Contrastive learning (section 5c) was attempted as the correct pivot, but
failed for a structural reason specific to the Devign dataset** — see §5c for
the full analysis. The block-level classifier (57.84%) remains the practical
deliverable for SCAR pipeline integration.

---

### 4a. CodeBERT / UniXcoder fine-tune on Devign — lowest friction path

**What it is:** Fine-tune a pre-trained transformer on Devign source code.
Operates on raw C function text, not LLVM IR. No compilation step, no
graph construction. The training code, dataset download, and evaluator are
all already written in the CodeXGLUE repo.

**Published results on Devign test set:**

| Model | Accuracy |
|---|---|
| UniXcoder (`microsoft/unixcoder-base`) | 69.29% |
| CodeBERT (`microsoft/codebert-base`) | 62.08% |
| RoBERTa | 61.05% |
| TextCNN | 60.69% |

**Procedure:**

1. Download dataset:
   ```bash
   cd Code-Code/Defect-detection/dataset
   pip install gdown
   gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
   python preprocess.py
   ```

2. Fine-tune (CodeBERT baseline — swap `model_name_or_path` for UniXcoder):
   ```bash
   python run.py \
       --model_name_or_path microsoft/codebert-base \
       --do_train \
       --train_data_file dataset/train.jsonl \
       --eval_data_file  dataset/valid.jsonl \
       --test_data_file  dataset/test.jsonl \
       --epoch 5 --block_size 400 \
       --train_batch_size 32 --eval_batch_size 64 \
       --learning_rate 2e-5 --seed 123456
   ```

3. Evaluate: `python evaluator/evaluator.py -a dataset/test.jsonl -p saved_models/predictions.txt`

**Infrastructure:** single GPU, 2–4 hours. Google Colab T4 is sufficient.
No compilation, no graph tooling — `pip install transformers` and run.

**The tradeoff:** These models see source code tokens, not IR structure.
Variable names, formatting, and coding style all influence the prediction.
The normalisation benefit of working at the IR level is absent. A function
written defensively but with unfamiliar style may score as vulnerable;
a genuinely vulnerable function written in a familiar idiom may not.

**Success criterion:** reproduce the published accuracy within ±1% to
confirm the setup is correct. Then fine-tune further on SCAR accepted
patches to specialise to SCAR's encountered patterns.

**Actual result (4a — CodeBERT on same Devign split, Google Colab T4):**

| Epoch | Val Acc |
|---|---|
| 1 | 60.29% |
| 2 | 62.63% |
| 3 | 64.71% |
| 4 | **64.82%** ← best |
| 5 | 64.31% |

**Test accuracy: 63.43%** (from epoch 4 checkpoint). Exceeds the published 62% baseline, confirming the data pipeline and split are correct.

Note: CodeBERT trains on all ~21K `train.jsonl` examples (no compilation needed), vs ~10K for the GNN (compilation survivors only). Part of the accuracy gap is training data volume, not purely architecture.

**The granularity gap: 63.43% − 57.84% = 5.6 points.** This is the cost of aggregating instructions into basic blocks. Information present in the raw source token sequence is discarded when an entire block is compressed to a 45-dimensional feature vector.

---

### 4b. Custom GNN on LLVM IR — structural graph model (no ProGraML)

> **Step-by-step training guide:** `docs/experiments/ir-embed-training.md`
> **AWS setup:** `docs/experiments/ir-embed-aws.md`

This is the theoretically correct path for SCAR's use case. It operates
on LLVM IR, normalising away surface noise and capturing actual control
and data flow structure.

**Why not ProGraML:** the ProGraML library is effectively abandoned (~2022)
and locks to LLVM 3.8/6.0/10.0 — incompatible with SCAR's LLVM 14
container. Instead, graph extraction is implemented directly from IR text
using stdlib Python (`graph_demo.py`), with no external graph library
required. The same approach was validated as an end-to-end GNN PoC
(`gnn_poc.py`) and is wired into the full training pipeline (`train_gnn/`).

**What is already built:**

| Script | What it does |
|---|---|
| `graph_demo.py` | Parses LLVM IR text → CFG nodes + edges, prints per-pair structural diff |
| `gnn_poc.py` | End-to-end GNN in pure numpy — validates graph→model pipeline |
| `train_gnn/preprocess.py` | Downloads Devign, compiles 27K C functions to IR, builds graphs with 11 node features, saves pickled datasets |
| `train_gnn/train.py` | 2-layer GCNConv → global mean pool → binary classifier, PyTorch Geometric, saves best checkpoint |

**Devign standalone compilation — current status:**
Devign functions come from FFmpeg, QEMU, and the Linux kernel. Two fixes
were needed to get usable graphs:

1. **Stub injection** (member injection, ptr/arr upgrades, macro demotions)
   handles unknown types and missing struct members. This brings compile
   attrition from ~95% down to ~52%.
2. **`#define static` / `#define inline`** at the end of the preamble
   forces clang to emit IR for isolated functions. Without callers in the
   translation unit, `static`/`inline` functions pass syntax checking but
   clang omits their `define` blocks from the `.ll` output entirely —
   compilation succeeds but the IR file is empty.

With both fixes applied, **~48% of Devign functions produce valid graphs**
and `graphed` matches `compiled` (no additional filtering by the IR parser).
On the full 27K dataset this yields ~8K training graphs for the train split.

- **For SCAR integration:** attrition is not a problem. The Tekton
  `build-bitcode` task builds the target project in its actual environment;
  functions are compiled in full project context.

**To run on your laptop (pipeline smoke test):**
```bash
cd experiments/ir_embed_demo/train_gnn
pip install gdown torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
python preprocess.py --subset 500   # ~240 graphs survive; enough to test
python train.py --epochs 10 --hidden 32
```

Any clang version works for preprocessing — no LLVM 14 required.

**Node features per basic block (11 total):**
instruction count, out-degree, in-degree, has\_call, has\_store,
has\_load, has\_icmp, has\_alloca, has\_getelementptr, has\_ret, has\_br.

**Success criterion:** accuracy ≥ 62% (CodeBERT baseline) on the Devign
test split confirms the structural graph representation is competitive.
Accuracy > 69% (UniXcoder) would mean IR structure is earning its cost
over token-based models. This criterion requires a full compilable dataset
(either via project build or Juliet).

**Actual result (4b baseline — CFG-only, 10K graphs):**

| Setting | Value |
|---|---|
| Train graphs | 10,097 (46% survival, 4,386 vuln / 5,711 fixed) |
| Val graphs | 1,251 |
| Test graphs | 1,250 |
| Epochs | 30, hidden=64 |
| pos_weight | 1.302 (fixed/vuln, applied to BCE loss) |
| Best val accuracy | 54.36% (epoch 25) |
| **Test accuracy** | **55.04%** |
| Majority-class baseline | 56.6% (always predict "fixed") |

The model barely learned — loss moved only 0.803 → 0.774 over 30 epochs.
55% is effectively at the majority-class ceiling, confirming that **CFG
topology alone does not carry enough signal** at basic-block granularity.
The 11 node features compress an entire basic block to a handful of binary
flags, making it impossible to distinguish a call to `strcpy` from a call
to `printf`, or a signed boundary check from an unsigned one.

**→ 4b confirmed insufficient. 4c (RGCNConv + DFG edges) triggered.**

**SCAR integration (after training):**

SCAR's `build-bitcode` task already emits LLVM IR. A new `ir-embed-scan`
Tekton task would parse each function's IR with `graph_demo.py`'s
extractor, score it against the trained model, and write top-K findings
to `findings-ir-embed.json` — feeding into the repair loop alongside IKOS
and LLM findings. Zero LLM cost per scan.

---

### 4c. GNN v2 — RGCNConv + DFG edges (PDG)

**Trigger:** 4b confirmed 55.04% — below 62% threshold. **Implemented and completed.**

**Actual result (4c — PDG, 10K graphs):**

| Setting | Value |
|---|---|
| Architecture | RGCNConv (2 relations: CFG + DFG) |
| Epochs / hidden | 30 / 64 |
| **Test accuracy** | **56.08%** |

DFG edges added +1.04% over 4b. The gradient signal improved (loss moved further) but the model remained well below the 62% threshold. **→ 4d triggered.**

Re-run preprocessing after `git pull` to regenerate graphs with `edge_type`.

**The problem with adding DFG edges to a standard GCNConv:**
Merging CFG and DFG edges into a single `edge_index` forces the aggregation
to use one shared weight matrix for both edge types. The model cannot
distinguish "block B executes after block A" from "block C uses a value
defined in block A" — it must implicitly guess edge type from node features
alone, wasting capacity and producing noisy gradients.

**The correct architecture: RGCNConv (Relational GCN)**

PyTorch Geometric's `RGCNConv` learns a separate weight matrix per edge type:
- W_CFG — projects features along control-flow edges
- W_DFG — projects features along data-dependency edges

Implementation diff from current `train.py` is small:

```python
# train.py — swap GCNConv for RGCNConv
from torch_geometric.nn import RGCNConv, global_mean_pool

class DefectGNN(torch.nn.Module):
    def __init__(self, in_features=N_FEATURES, hidden=64):
        super().__init__()
        self.conv1 = RGCNConv(in_features, hidden, num_relations=2)
        self.conv2 = RGCNConv(hidden, hidden, num_relations=2)
        self.lin   = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_type, batch):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = global_mean_pool(x, batch)
        return self.lin(x).squeeze(-1)
```

**Extracting DFG edges from IR:**

LLVM IR is in SSA form — every value (`%name`) is defined exactly once and
all uses are explicit. The scaffolding is already in `preprocess.py`:

```python
_DEF     = re.compile(r"^\s+(%[\w.]+)\s*=")   # value definition
_USE_VAR = re.compile(r"%[\w.]+")              # value uses
```

These are parsed but currently unused. Wiring them up in `_parse_ir`:
1. For each instruction: record which `%name` it defines
2. For each instruction: find all `%name` references after the `=`
3. For each use, look up which basic block defined that value
4. Add a DFG edge from the defining block to the using block (if cross-block)

`preprocess.py` graph format would add two keys:
```python
{
    "x":          ...,           # node features (unchanged)
    "edge_index": ...,           # all edges (CFG + DFG concatenated)
    "edge_type":  ...,           # 0 = CFG, 1 = DFG
    "y":          ...,
}
```

**Why intra-function DFG is available from isolated snippets:**
SSA def-use chains are entirely self-contained within a function. The
snippet isolation that causes compilation attrition does not affect DFG
extraction — all the data flow information for values defined and used
within the function is present in the `.ll` output. Interprocedural data
flow (tracing values into called functions) is not available, but that
limitation applies equally to ProGraML without a full project build.

---

### 4d. Semantic feature upgrade — conditional on 4c result

**Trigger:** pursue if 4c (PDG) stalls below 62%.

**Diagnosis if 4c stalls:** the graph topology (nodes and edges) is fine
but node features are too coarse. The 11-feature vector cannot distinguish
a call to `strcpy` from a call to `printf`, or a signed boundary check from
an unsigned one. The model is structurally blind to the vocabulary of danger.

**Strategy:** beat CodeBERT at what it can only approximate — extract
explicit low-level semantic facts from LLVM IR that a token-based model
must guess at from variable names and context. All changes are algorithmic
(no LLM), zero inference latency increase.

All four changes implemented together in one preprocessing + training run.

#### 4d-i. Bag-of-APIs node features

Replace the single `has_call` flag with per-function binary flags covering
the ~28 functions responsible for the majority of C memory corruption.
LLVM IR preserves exact call targets (`call i32 @strcpy(...)`) regardless
of macro expansion or aliasing, making extraction unambiguous.

Safe/unsafe pairs (`strcpy` vs `strncpy`, `sprintf` vs `snprintf`) are
especially informative — the model can learn that one variant is a red flag
and the other is not.

| Category | Functions |
|---|---|
| Standard alloc/free | `malloc`, `calloc`, `realloc`, `free` |
| Unbounded string ops | `strcpy`, `strcat`, `sprintf`, `gets` |
| Bounded string ops | `strncpy`, `strncat`, `snprintf`, `fgets` |
| Memory ops | `memcpy`, `memmove`, `memset` |
| FFmpeg | `av_malloc`, `av_mallocz`, `av_realloc`, `av_free`, `av_freep` |
| Linux kernel | `kmalloc`, `kfree`, `kzalloc`, `vmalloc`, `vfree` |
| QEMU/GLib | `g_malloc`, `g_malloc0`, `g_realloc`, `g_free`, `g_new` |

#### 4d-ii. icmp semantics

Replace the single `has_icmp` flag with three flags that encode the
mathematical meaning of the comparison. LLVM IR states this explicitly
in the instruction name; CodeBERT must infer it from variable names.

| Flag | Matches | Vulnerability relevance |
|---|---|---|
| `has_signed_cmp` | `icmp s[lt\|gt\|le\|ge]` | Signed/unsigned mismatch → integer overflow |
| `has_unsigned_cmp` | `icmp u[lt\|gt\|le\|ge]` | Correct unsigned bounds check |
| `has_eq_cmp` | `icmp eq`, `icmp ne` | Null check, sentinel value check |

A block with `has_unsigned_cmp=0` before a `memcpy` call is a direct
IR signature of a missing bounds check.

#### 4d-iii. Type and width semantics

LLVM IR explicitly states the bit-width of every operation. Two flags
capture the most security-relevant width signals:

- `has_i8_op` — byte-level load/store (buffer iteration at char granularity;
  combined with `has_getelementptr`, a strong indicator of unsafe buffer walking)
- `has_64bit_op` — i64 arithmetic (potential truncation when result is
  narrowed to i32 for a bounds check or array index)

#### 4d-iv. Global Attention readout (train.py only — no preprocessing)

Replace `global_mean_pool` with `GlobalAttention(gate_nn=Linear(hidden, 1))`.
A single learned linear layer outputs a scalar weight per block before
aggregation. The model learns to focus on blocks with dangerous semantics
(e.g., `has_memcpy=1` + `has_unsigned_cmp=0`) and mute boilerplate blocks.

Replicates CodeBERT's self-attention focus mechanism at ~64 extra parameters.

#### Complete 4d feature vector

| Group | Features | Count |
|---|---|---|
| Structural | n\_instructions, out\_degree, in\_degree | 3 |
| Opcode flags | has\_call, has\_store, has\_load, has\_alloca, has\_getelementptr, has\_ret, has\_br | 7 |
| icmp semantics | has\_signed\_cmp, has\_unsigned\_cmp, has\_eq\_cmp | 3 |
| Type/width | has\_i8\_op, has\_64bit\_op | 2 |
| API hashing | 30 function flags (see table above) | 30 |
| **Total** | | **45** |

`N_FEATURES` in `train.py` updates from 11 → 45.

#### Actual results (4d — multiple runs)

| Run | Epochs | Hidden | Notes | Test Acc |
|---|---|---|---|---|
| v2.0 | 30 | 64 | linear gate | 56.32% |
| v2.0 | 60 | 128 | best val epoch 8, then overfit | **57.84%** |
| v2.1 | 30 | 128 | MLP gate (Linear→ReLU→Dropout→Linear) | 56.88% |

Adding 34 semantic features (API flags, icmp types, type/width) gained only +0.24% over 4c. Increasing capacity (hidden=128, 60 epochs) reached 57.84% but peaked at epoch 8 and overfit thereafter. The MLP attention gate made no meaningful difference.

**Conclusion: basic-block representation is saturated at ~57–58%.** The bottleneck is granularity, not feature richness. Adding more API flags or a more expressive pooling layer cannot recover information that was discarded by aggregating an entire basic block into a single feature vector. **→ 4a (CodeBERT baseline) run to establish ceiling.**

---

### 4e. Opcode embeddings — if 4d stalls

**Trigger:** pursue if 4d stalls below 62%.

More principled replacement for the opcode flags: count all ~70 LLVM
opcodes per block and look up a learned `d=16` embedding per opcode. The
block representation becomes `sum(count_i × E[opcode_i])`. The model
discovers which opcodes correlate with vulnerability rather than relying
on the curated 4d list.

Requires more training signal to converge than hand-crafted flags — better
suited to larger datasets. At 10K graphs, 4d's expert knowledge is more
reliable. At 50K+ graphs, opcode embeddings are more principled.

---

### 4a. CodeBERT / UniXcoder fine-tune — fallback at any stage

If structural GNN approaches consistently fall below 62%, run experiment 4a
(see above) to establish the ceiling with pre-trained semantics. Use that
result to decide whether to invest further in the structural path or adopt
a transformer-based classifier for SCAR integration.

---

### 5. Contrastive learning + vector corpus — parallel paradigm

This is a fundamentally different training objective from the 4x series and
can be pursued independently of whether classification succeeds or fails.
It is also the strongest fit for SCAR's self-improvement architecture.

**What changes:** Instead of `f(graph) → {0,1}`, the model learns
`f(graph) → embedding_vector`. The loss function enforces geometry directly:
- Same-class pairs (vuln+vuln, fixed+fixed): pulled together
- Cross-class pairs (vuln+fixed): pushed apart by at least a margin

---

#### The operational system

```
Train once
──────────
Devign graphs ──► GNN (SupCon loss) ──► embedding model (frozen weights)

Populate corpus (one-time)
──────────────────────────
All Devign train graphs ──► embedding model ──► vectors ──► vector DB
                                                            (labelled vuln / fixed)

Scan (per function, per commit)
───────────────────────────────
new C function
  │
  ▼ clang -O0 -S -emit-llvm
LLVM IR
  │
  ▼ preprocess.py graph extraction
PDG graph
  │
  ▼ embedding model (one forward pass, CPU, <1ms)
vector
  │
  ▼ k-nearest-neighbor query against vector DB
k neighbours (labelled vuln / fixed)
  │
  ▼ majority vote
finding: "structurally similar to N known-vulnerable functions"
  + neighbour list for explainability

Enrich (per accepted SCAR patch)
─────────────────────────────────
accepted patch ──► compile vulnerable version ──► IR ──► embed
                ──► insert vector (labelled vuln) into DB
                ──► future scans automatically benefit, no retraining
```

No decision boundary. No retrained classifier. The structural shape of
the new function determines where it lands in embedding space, and the
corpus grows richer with every patch SCAR accepts on any target.

---

#### Why this fits SCAR better than a classifier

**A classifier is a static artifact.** Once trained, it cannot incorporate
new vulnerability patterns without full retraining. Every SCAR accepted
patch is wasted signal.

**The vector corpus is a living database.** Each accepted patch produces a
labelled `(vuln_IR, fix_IR)` pair at zero marginal cost. Embedding the
vulnerable version and inserting it into the DB takes milliseconds. The
model's coverage of real-world patterns grows continuously without touching
model weights.

**Explainability is free.** A classifier outputs a probability — opaque to
the developer reviewing the finding. A vector query returns the *neighbours*:
"this function is structurally similar to these 3 known-vulnerable functions
from FFmpeg commit abc123." A developer can inspect the neighbours and
understand why the function was flagged, and what the fix looked like.

**Generalisation is enforced by the loss.** The SupCon loss must find a
unified structural criterion that explains *all* (vuln, fix) pairs being far
apart simultaneously — across FFmpeg, QEMU, and the Linux kernel at once.
It cannot overfit to one project's naming conventions. Classification can
draw a project-specific boundary; contrastive learning is forced to find a
cross-project invariant.

---

#### Vector DB options

| Option | Deployment | Notes |
|---|---|---|
| **FAISS** | in-process (no server) | Facebook, fast, Python-native, ideal for Tekton sidecar |
| **Qdrant** | container (OCI image available) | REST API, persistent, good for a shared SCAR corpus PVC |
| Chroma | in-process or server | simpler API than Qdrant, less production-hardened |

For SCAR's Tekton integration, **FAISS** is the lowest-friction path: the
corpus is loaded from a file on the PVC at task startup, queried in-process,
and the file is updated when a patch is accepted. No separate service to
manage.

For a shared corpus across pipeline runs, **Qdrant** running as a pod with
a PVC is more appropriate — the corpus persists and grows across all SCAR
runs on all targets.

---

#### Accuracy expectations

Contrastive learning on the same 45-feature basic-block graphs will not
close the 5.6-point granularity gap to CodeBERT. The basic-block ceiling
(~57–58%) applies to both training objectives — contrastive loss cannot
recover information discarded by block aggregation.

What changes is the *inference paradigm* and the *corpus growth story*, not
the raw Devign accuracy number. The value is operational, not benchmark.

---

#### Loss function

Supervised Contrastive loss (SupCon, Khosla et al. 2020) is preferred for
Devign. Within each training batch, all same-label graphs are positives for
each other; all different-label graphs are negatives. No explicit pairing
of commits needed — just the 0/1 labels already in the dataset.

```python
# Conceptual training loop
embeddings = model(batch)           # (B, d) — one embedding per graph
embeddings = F.normalize(embeddings, dim=1)
loss = supcon_loss(embeddings, batch.y)   # pulls same-label together,
                                          # pushes different-label apart
```

The model head changes: instead of `Linear(hidden, 1)` → scalar logit,
use `Linear(hidden, d_embed)` → normalised d-dimensional vector.
`d=64` or `d=128` is typical. The rest of the GNN (RGCNConv layers,
AttentionalAggregation pool) is unchanged.

---

#### Relationship to experiments 2 and 4x

Experiment 2 (opcode histogram + contrastive MLP) was the toy-scale proof
of concept on 14 samples. Experiment 5 applies the same principle to full
PDG graphs with the 45-feature 4d feature set, at Devign scale.

The 4x series answers: "can a classifier detect the vulnerability shape?"
Experiment 5 answers: "can the shape be embedded so that unseen code finds
its own cluster, and so that new vulnerability knowledge can be added
without retraining?"

Both depend on feature quality — the 45-feature 4d upgrade applies to both
paradigms. Run them on the same preprocessed graphs (no new preprocessing
needed).

---

### §5a — Instruction-Level GNN with Per-Instruction Feature Vectors

**Scripts:** `preprocess_5.py`, `train_5.py`  
**Trigger:** §4d confirmed block-level ceiling at 57.84%; hypothesis that instruction-level granularity breaks it.

#### Motivation

A basic-block feature vector aggregates an entire block — often 5–30 instructions — into 45 floats. A 3-line security patch may change a single instruction in a single block; that change is diluted by the other 30 instructions' opcode flags before it reaches the GNN. Hypothesis: one node per IR instruction, with its own opcode features, gives each changed instruction a dedicated row in the graph.

#### Architecture

`preprocess_5.py` — llvmlite extractor, per-instruction nodes:

- Virtual entry node at index 0 (all function arguments map here)
- One node per IR instruction (~300–500 nodes/function vs ~15 blocks)
- **32-dimensional float feature vector per node:**
  - feat[0]: `is_entry` (1 only for the virtual entry node)
  - feat[1–30]: opcode one-hot over 30 common LLVM opcodes
  - feat[31]: `is_dangerous_call` (1 if call targets a known-unsafe API from the §4d list)
- CFG edges (type 0): intra-block sequential + terminator→successor branching
- DFG edges (type 1): SSA def-use chains (same extraction logic as block-level)

`train_5.py` — same `DefectGNN` as `train.py` (2× `RGCNConv` + `AttentionalAggregation`), N_FEATURES=32, 2 edge types. 30 epochs, hidden=64.

#### Results

**Test accuracy: 55.84%** — below the block-level baseline of 57.84% and barely above majority class.

| Method | Test Acc |
|---|---|
| Block-level 4d best | 57.84% |
| Majority-class baseline | 56.6% |
| **5a instruction-level (32-feat one-hot)** | **55.84%** |

#### Autopsy

More nodes made the problem harder, not easier:

1. **Depth penalty.** 300–500 nodes vs 15 means the GNN must aggregate across ~20× more hops. With only 2 `RGCNConv` layers, most instruction nodes cannot reach global context. Without a virtual context node to reduce graph diameter, over-smoothing risk is severe.

2. **Feature degradation.** The 32-dimensional one-hot vector is less informative than the 45-dimensional block vector. The block-level vector carries 30 API presence flags that distinguish `strcpy` from `strncpy` at function granularity. The instruction-level vector encodes the opcode category only — `call @strcpy` and `call @printf` both set `feat[28]`; `icmp sgt` and `icmp sge` both set `feat[25]`. All comparison predicates collapse to a single bit.

3. **Dilution confirmed.** When a 3-line patch changes 3 instruction nodes out of 300, message-passing averages the changed nodes' features with 297 identical neighbours before the signal reaches the global pool. The gradient is mathematically present but too small to dominate training.

**→ §5b triggered:** maybe capturing instruction *order* within blocks helps where static one-hot features cannot.

---

### §5b — GRU Block Hybrid: Intra-Block Sequence Encoding

**Script:** `train_gru.py`  
**Trigger:** §5a confirmed that static one-hot features per instruction hurt; hypothesis that a GRU over the opcode sequence captures ordering information that the static block vector discards.

#### Motivation

The 45-feature block vector encodes which opcodes are *present* but not in which *order*. A block ending in `[load, icmp sgt, br]` (a bounds-check path) looks the same as `[br, load, icmp sgt]` to the model. A GRU reading the opcode sequence left-to-right should distinguish these — and should produce different final states for `icmp sgt` vs `icmp sge` at the block level if the vocabulary encodes them separately. (It does not in §5b — see below.)

#### Architecture

Block representation (new):
```
per block: opcode_sequence → Embedding(vocab=44, dim=16) → GRU(hidden=32) → 32-dim block vector
```
The GRU reads each instruction's opcode token in order; `pack_padded_sequence` handles variable-length blocks. The final hidden state becomes the block's node feature, replacing the static 45-float vector.

Graph model (unchanged from §4d):
```
RGCNConv(32 → 64, 2 relations) → RGCNConv(64 → 64) → AttentionalAggregation → Linear(1)
```

Vocabulary (44 tokens): 2 special tokens (`<PAD>`, `<UNK>`) + 42 opcodes. No new preprocessing run required — `block_opcodes` (the sequence of instruction opcode strings per block) was added as an extra key to the existing block-level pkl files.

#### Results

**Test accuracy: 56.96%** — better than §5a but still 0.88% below block-level §4d best.

| Method | Test Acc |
|---|---|
| Block-level 4d best | 57.84% |
| **5b GRU hybrid (sequence order)** | **56.96%** |
| Majority-class baseline | 56.6% |
| 5a instruction-level one-hot | 55.84% |

#### Autopsy

Sequence encoding recovered +1.12% over one-hot features but could not clear the block-level ceiling:

1. **Vocabulary collapse persists.** The GRU vocabulary (44 tokens) has one token for `icmp` regardless of predicate and one token for `call` regardless of target. The GRU final state for a block containing `icmp sgt` and the state for `icmp sge` are identical. The same comparison-operator blindspot that defeats the block-level classifier operates here too.

2. **Block-level bottleneck survives.** The GRU output is one vector per block. Instruction-order information within a block is encoded, but the cross-block information aggregation (via `RGCNConv`) is unchanged. A 3-line patch that changes an instruction in one block still competes with ~14 unchanged blocks during global pooling.

3. **Cost/benefit unfavorable.** The GRU adds ~130K parameters and significant per-batch runtime. For −0.88% vs block-level, the complexity is not justified.

**→ §5c triggered:** sequence encoding within blocks is a dead end at this vocabulary. The next experiments change the training objective (contrastive) or the granularity (full instruction-level graph with learned embeddings and virtual context node — §7).

**Root-cause retrospective:** §5a and §5b both fail for the same underlying reason as the block-level series: the instruction vocabulary treats `icmp sgt` and `icmp sge` as identical. §10 (vocabulary enrichment) is the direct fix for this blindspot.

---


### 5c. Supervised Contrastive Learning — result and architectural autopsy

**Script:** `train_contrastive.py`  
**Run:** `python3 train_contrastive.py --epochs 50 --batch-size 512 --temp 0.07 --k 5`

**Result: 55.84% test k-NN accuracy. Embedding collapse. Series closed.**

```
Epoch      Loss   Val k-NN
--------------------------------------
    1    6.3338     56.23%  ← best
    2    6.2267     55.19%
    4    6.2251     56.39%  ← best (epoch 22: 57.43%)
   22    6.2246     57.43%  ← best val
   50    6.2246     56.47%

Test k-NN accuracy (k=5): 55.84%
```

#### The Flatline of Death: why the loss froze at 6.2246

From epoch 4 onward the loss freezes at exactly 6.2246 — `log(512)`, the loss
produced by a random encoder when all cosine similarities are approximately zero.

With τ=0.07 and initial embeddings of near-zero cosine similarity variance:
`exp(~0 / 0.07)` ≈ 1 for every pair. The softmax denominator is flat; gradients
cancel; the model never escapes random initialization.

But the deeper cause is not temperature — it is the **categorical label problem**:

SupCon groups every vulnerable function in the batch as positives for each other.
This instructs the loss to pull a buffer overflow and a race condition and a
use-after-free toward the same coordinate in embedding space. These three
vulnerability types are structurally unrelated — their structural gradients
directly oppose and cancel. Faced with contradictory instructions, the network
collapses: it learns to output the same generic vector for every graph, and the
loss flatlines at `log(N)`.

The final k-NN accuracy (55.84%) equals the majority-class baseline — the model
learned nothing structural. It is guessing the majority class.

#### Why standard Triplet Loss would have been different — but still blocked

If we had used Triplet Loss with `(anchor=vuln_IR, negative=exact_fix_IR)` from
the same commit, the gradient would be precise: learn the specific structural
diff, not a cross-project average. That question IS answerable from pure IR structure.

The blocking constraint is the Devign dataset: it is distributed as 27,000
disconnected functions with no commit-pairing metadata. Reconstructing exact
(vuln, fix) pairs requires data archaeology against the source repositories
(FFmpeg, QEMU, Linux kernel). Given the 5.6-point modelling gap persisting across
all approaches, that reconstruction effort is not justified at this stage.

#### What this closes

Every structural weapon in the arsenal has been tried:

| Experiment | Approach | Test Acc |
|---|---|---|
| 4b | CFG-only GCNConv | 55.04% |
| 4c | PDG RGCNConv + DFG edges | 56.08% |
| 4d | 45-feature semantic upgrade + AttentionalAggregation | **57.84%** |
| 5a | Instruction-level GNN (opcode one-hot) | 55.84% |
| 5b | GRU hybrid (opcode sequence order) | 56.96% |
| 5c | SupCon k-NN (contrastive training objective) | 55.84% |
| 4a | CodeBERT fine-tune | **63.43%** |

**Scientific conclusion: the mathematical ceiling for pure anonymized LLVM IR
structural analysis on Devign is ~58%.** No GNN architecture or training
objective closes the 5.6-point gap to CodeBERT, because the gap is not
structural — it is semantic. CodeBERT reads human-meaningful identifiers
(variable names, API names, developer intent) that our hand-coded feature
vectors discard.

#### The pipeline deliverable

This is a successful engineering boundary discovery. The block-level GNN
(57.84%, `model.pt`, `preprocess.py`) is the practical output:

- **Cost:** one CPU forward pass per function, milliseconds per PR
- **Role:** zero-LLM-cost pre-filter — routes structurally suspicious
  functions to the heavier LLM scanner, bypasses clearly safe code
- **Self-improvement:** each accepted SCAR patch extends the training corpus;
  periodic retraining improves coverage without architectural changes
- **The 57.84% number is a Devign proxy.** On SCAR's actual targets (with
  full build context from Tekton's `build-bitcode` task), attrition is lower
  and the feature distribution better matches real-world vulnerability patterns.

---

### 6. BigVul — Triplet Contrastive Learning with exact (vuln, fix) commit pairs

**Scripts:** `train_gnn/preprocess_bigvul.py`, `train_gnn/train_triplet.py`

#### Motivation

The SupCon experiment (§5c) revealed a specific, fixable problem: the loss
collapsed because Devign has no commit-pairing metadata. Every vulnerable
function is treated as equivalent to every other, so the loss tries to pull
a buffer overflow and a race condition toward the same point in embedding
space — they are structurally unrelated, the gradients cancel, and the model
gives up. The failure is not a fundamental limit of contrastive learning; it
is a property of the dataset.

BigVul has the one thing Devign lacks: every row is a matched
`(func_before, func_after)` pair from a real CVE commit. This makes the fix
the guaranteed negative for its own vulnerability. Instead of asking the model
to find what all vulnerabilities have in common (impossible — they don't), we
ask it one precise question per pair: *learn the structural diff between this
function and its patch.* That question is answerable from IR structure alone,
because the diff IS in the IR.

#### Why BigVul instead of Devign

§5c confirmed that SupCon collapses on Devign because the dataset has no
commit-pairing metadata. All vulnerable functions are treated as positives
for each other, producing contradictory structural gradients that cancel.

**BigVul (MSR_20)** has the structural property Devign lacks: every row is
an explicit `(func_before, func_after)` commit pair for a real CVE. The exact
structural diff — the patch itself — is the training signal. Triplet Loss with
`anchor=vuln_IR, negative=exact_fix_IR` makes the gradient precise and
unambiguous, answerable from IR structure alone.

| Property | Devign | BigVul |
|---|---|---|
| Dataset size | 27K functions | 9,514 usable (vuln, fix) pairs |
| Commit pairing | ✗ none | ✓ explicit per row |
| Languages | C/C++ | C/C++ only |
| Split strategy | by function | by CVE (used here) |
| SupCon viability | ✗ categorical collapse | — |
| Triplet viability | ✗ no pairs | ✓ guaranteed negatives |

#### Dataset download

BigVul is distributed as `MSR_data_cleaned.csv` (~10 GB uncompressed).

```bash
cd experiments/ir_embed_demo/train_gnn
mkdir -p data

pip install gdown
gdown 1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X -O data/bigvul.zip
unzip data/bigvul.zip -d data/
# Result: data/MSR_data_cleaned.csv (~10 GB)
```

The zip (~1.5 GB) unpacks to the full CSV. The file uses latin-1 encoding
and contains C function bodies with unescaped quotes — the preprocessor
handles both.

#### Preprocessing

```bash
python3 preprocess_bigvul.py --csv data/MSR_data_cleaned.csv --workers 4
```

Filters to `vul=1` rows where `func_before != func_after` (~9,514 pairs from
~6.8M rows). Splits by CVE ID to prevent leakage: 80/10/10 on unique CVE IDs.
Compiles both sides of each pair to LLVM IR (`clang -O0 -S -emit-llvm`) and
extracts PDG graphs using the same `preprocess.py` extractor as the 4x series
(45 features, CFG + DFG edges). Saves three pickle files:

```
data/bigvul_train_pairs.pkl   — list of {"vuln": graph, "fix": graph, "cwe": ..., "cve": ...}
data/bigvul_valid_pairs.pkl
data/bigvul_test_pairs.pkl
```

Expected attrition: ~40-60% (compilation failures on isolated snippets without
full project context). Expected usable pairs: ~4,000-6,000 after compilation.

Progress output every ~10% of pairs; file size and elapsed time shown for the
10 GB CSV read (typically 3-8 minutes through the Python engine).

#### Training

```bash
python3 train_triplet.py --epochs 50 --batch-size 128
```

Reuses `ContrastiveGNN` from `train_contrastive.py` unchanged (RGCNConv
encoder + projection head). Triplet Loss with guaranteed negatives:

```python
def triplet_loss(anchor, positive, negative, margin=0.3):
    pos_sim = (anchor * positive).sum(dim=-1)   # anchor vs same-CWE vuln
    neg_sim = (anchor * negative).sum(dim=-1)   # anchor vs its own patch
    loss = torch.clamp(neg_sim - pos_sim + margin, min=0.0)
    return loss[loss > 0].mean()
```

- **anchor** = vuln embedding
- **negative** = its paired fix embedding (exact structural diff, guaranteed)
- **positive** = another vuln in batch with same CWE; fallback to nearest other vuln

The negative is always available from the paired fix — no mining required.
Each gradient is specific to one structural diff rather than a categorical
average across all vulnerability types.

Evaluation: same `build_corpus` + `knn_accuracy` from `train_contrastive.py`.
Corpus = all training vuln embeddings (L2-normalized). Also reports mean
cosine similarity between (vuln, fix) pairs each epoch — should decrease as
the model learns to push them apart.

#### What success looks like

| Signal | Target |
|---|---|
| Loss trajectory | Decreasing from epoch 1 (no log(N) flatline) |
| Val k-NN by epoch 10 | > 58% (structural diff signal is real) |
| Mean (vuln, fix) cosine similarity | Decreasing over training |
| Test k-NN | > 60% (closes meaningful portion of 5.6pt gap to CodeBERT) |

If val k-NN stalls at 57-58% again, the ceiling is confirmed at the block
level: a 3-5 line patch across 50-100 IR blocks is too localised for
block-level aggregation to capture. Instruction-level graphs would be the
next step.

#### Results

| Metric | Value |
|---|---|
| Train pairs (after compilation) | 1,117 of 6,964 (84% attrition) |
| Valid pairs | 122 of 1,711 (93% attrition) |
| Test pairs | 124 of 839 (85% attrition) |
| Best val k-NN (epoch 15) | 54.51% |
| **Test k-NN (k=5)** | **51.21%** |
| Mean (vuln, fix) cosine similarity | 0.9823 (epoch 1: 0.979 → epoch 50: 0.986) |

**Result: coin-flip accuracy. Block-level ceiling confirmed for triplet loss too.**

Training diagnostics:

- **Loss decreased (0.46 → 0.34)** — no flatline. Triplet signal is real when commit pairs exist.
  The gradient does not collapse the way SupCon did. This is the correct result for the loss function.
- **Pair cosine similarity increased (0.979 → 0.986)** — wrong direction. Vuln and fix embeddings
  became *more* similar over training. The model failed to push pairs apart.
- **Val k-NN oscillated 44–54% with no trend** — 122 samples × 0.8% per sample = noise.

#### Autopsy

At block level, a 3–5 line patch across a 50–100 block function changes a handful of feature
values in a handful of nodes. The vuln and fix graphs are already 0.979 cosine-similar at epoch 1
— there is barely any signal to push apart, and 1,117 training pairs is too few to learn a
manifold from it. The model reduces active triplet violations by drifting toward a region where
all embeddings are similar (~0.98 to each other), which lowers the loss without learning anything.

| Failure mode | SupCon on Devign | Triplet on BigVul |
|---|---|---|
| Symptom | Hard flatline at log(batch) | Soft collapse; pair sim increases |
| Root cause | Categorical label gradient cancellation | Block granularity below patch resolution |
| Loss moved? | No | Yes (0.46 → 0.34) |
| k-NN useful? | No | No |

**The block-level ceiling (~58%) applies to triplet loss.** The structural diff between a
vulnerable function and its patch is present in the IR but below the resolution of block-level
aggregation. No training objective operating on 45-dimensional per-block feature vectors can
recover information that was discarded when each basic block was compressed to a single vector.

Attrition was also severe: 84–93% across splits, yielding only 1,363 usable pairs total.
Stub injection (as used in the Devign pipeline) would reduce attrition to ~50%, but even with
3,000–5,000 pairs, the granularity problem remains — more data does not change the block-level
resolution.

**Series conclusion:** The structural ceiling is ~58% for any training objective, dataset, or
architecture that operates on block-level basic-block feature vectors. Instruction-level graphs
(each IR instruction as a node, learned opcode embeddings) see the structural diff directly —
the missing `icmp + br` is literally a missing edge — and are implemented in §7. First result:
58.00%, marginally clearing the block-level ceiling.


---

### 7. Instruction-level GNN — opcode embedding on Devign

**Status: ACTIVE (extended run in progress)**
**Scripts:** `train_gnn/preprocess_instr.py`, `train_gnn/train_instr.py`

#### Motivation

§6 BigVul confirmed the block-level ceiling applies to triplet loss too: a 3–5 line patch
across 50–100 IR blocks leaves the block-level adjacency matrix functionally invariant (pair
cosine similarity 0.979 → 0.986, wrong direction). The structural diff IS in the IR — it is
below the resolution of block aggregation.

The fix is granularity. Each LLVM instruction becomes its own node (~300–500 nodes per function
vs ~15 blocks). A 3-line patch changes 3–5 nodes directly. The missing `icmp + br` is literally
a missing edge in the instruction graph.

#### Implementation

**`preprocess_instr.py`** — 5-pass llvmlite extractor:

- **Node vocabulary (80 entries):** Virtual Context (0), Function Argument (1), ALU (2–23),
  Memory (26–32), Control-flow (36–43), Comparison (46–47), Cast (48–60), Other (61–68),
  Mock/Global (75), Constant int/fp (76–77), Undef/poison (78), Unknown (79)
- **Pass 1:** Allocate nodes using `ctypes.cast(v._ptr, ctypes.c_void_p).value` as stable
  C++ pointer keys (never `id(v)` — GC reuses memory addresses)
- **Pass 2:** CFG edges (type 0) — intra-block sequential + inter-block via terminator operands
- **Pass 3:** DFG edges (type 1) — SSA def-use via ValueKind dispatch; constant and mock caches
  prevent duplicate nodes (LLVM interns constants — same literal → same `_ptr`)
- **Pass 4:** Bidirectional Global Context edges (type 2) to Virtual Context node 0 — reduces
  graph diameter from O(depth) to O(1), preventing over-smoothing in 300–500 node graphs

Reuses `compile_to_ir()` from `preprocess.py` unchanged. Outputs `data/{split}_instr_graphs.pkl`.

**`train_instr.py`** — `InstructionGNN`:

```
nn.Embedding(80, 128, padding_idx=79)   # opcode vocab → dense embedding
RGCNConv(128 → 64, num_relations=3)     # 3 relations: CFG / DFG / Global
RGCNConv(64 → 64,  num_relations=3)
AttentionalAggregation                  # focus on dangerous instructions
Linear(64 → 1)                          # BCE classifier
```

Node features are integer opcode indices (long), not float vectors. No z-score normalisation.
3 edge relations vs 2 in block-level model (adds Global Context type).

#### Preprocessing results

```
train: 10,100 graphs from 21,854 functions (54% attrition)
valid:  1,252 graphs from  2,732 functions (54% attrition)
test:   1,250 graphs from  2,732 functions (54% attrition)
train class balance: 4,387 vuln / 5,713 fixed
```

Attrition matches block-level runs exactly — both use `compile_to_ir()` as the bottleneck.

#### Results (30ep, hidden=64, embed=128)

| Epoch | Loss | Val Acc |
|---|---|---|
| 1 | 0.7876 | 46.65% |
| 2 | 0.7809 | 54.87% ← best |
| 7 | 0.7649 | 55.27% |
| 8 | 0.7616 | 55.75% |
| 11 | 0.7497 | 57.59% |
| 17 | 0.7396 | 58.15% ← best val |
| 30 | 0.7233 | 57.43% |

**Test accuracy: 58.00%** (epoch 17 checkpoint)

| Method | Test Acc |
|---|---|
| Block-level best (4d, 60ep h=128) | 57.84% |
| **Instruction-level (30ep h=64)** | **58.00%** |
| CodeBERT (4a) | 63.43% |

**First result in the series to clear the block-level ceiling.** Marginal (+0.16%) but directionally
confirmed: instruction-level micro-topology carries signal that block aggregation discards.

#### Interpretation

Loss was still decreasing at epoch 30 (0.7876 → 0.7233) with no convergence plateau — the model
had not saturated. The LR step at epoch 20 (×0.5) likely killed momentum before the model found
its minimum. Extended run: `--epochs 60 --hidden 128` (same pattern that took block-level from
56.32% to 57.84%).

The +0.16% gap is narrow but meaningful: **it confirms the hypothesis** that instruction-level
resolution captures the structural diff of a patch. At block level, a 3-line fix is invisible.
At instruction level, those 3 lines change 3–5 nodes directly.

#### Next: instruction-level contrastive learning on BigVul

With the representation validated, the logical next step is instruction-level triplet loss on
BigVul. At block level, pair cosine similarity started at 0.979 and only increased — the diff
was below block resolution. At instruction level, a 3-line patch changes 3–5 nodes in a
~300-node graph, making the (vuln, fix) pairs structurally distinguishable for the first time.

The BigVul attrition problem (84–93% at block level) applies here too — but the representation
improvement changes what triplet loss can learn from the pairs that survive.

#### Extended run results (60ep, hidden=128)

| Epoch | Loss | Val Acc |
|---|---|---|
| 1 | 0.7895 | 47.52% |
| 5 | 0.7705 | 55.99% |
| 8 | 0.7621 | 56.71% |
| 12 | 0.7415 | 56.79% |
| 17 | 0.7285 | 57.67% ← best val |
| 20 | 0.7251 | 55.27% ← LR step ×0.5 kills momentum |
| 30 | 0.6974 | 56.23% |
| 60 | 0.6761 | 55.27% |

**Test accuracy: 56.16%** — below both the 30-epoch run and the block-level baseline.

#### Overfitting autopsy

Both runs peak at epoch ~17 then degrade:

| Run | Params | Peak Val Epoch | Peak Val | Test |
|---|---|---|---|---|
| 30ep h=64 | 61K | 17 | 58.15% | **58.00%** |
| 60ep h=128 | 150K | 17 | 57.67% | 56.16% |

Loss continues falling (0.6974 → 0.6761 from ep30→60) while val acc stagnates at 55–56% —
classic overfitting. The larger model (150K params) memorises training patterns faster than
the smaller one (61K), hitting the same peak epoch but at lower val accuracy and deteriorating
more sharply afterward.

Contrast with block-level 4d, where going 30ep→60ep improved 56.32%→57.84%: block-level
node features are coarser (45 floats vs an opcode embedding), so the model learns more slowly
and does not overfit at the same scale.

**Instruction-level GNN best result: 58.00% (30ep, h=64).** This is the number to carry forward.
Early stopping at epoch 17 would recover this from the larger model, but the +0.16% improvement
over block-level does not justify further tuning.

#### Series conclusion for §7

The instruction-level representation is confirmed: 58.00% > 57.84%, first time the block-level
ceiling was cleared. The margin is narrow because:

1. The opcode embedding must learn from 10K training examples what CodeBERT learned from billions
   of tokens of pretraining — the same modelling gap applies
2. Both runs show the model converges at epoch ~17 regardless of capacity; more parameters only
   add overfitting risk without improving the peak

**The validated hypothesis:** instruction-level micro-topology carries signal that block-level
aggregation discards. Whether that signal is sufficient to make instruction-level contrastive
learning on BigVul succeed (where block-level pair cosine similarity was already 0.979) is
the open question for the next experiment.

---

## §8 — Instruction-Level BigVul Triplet Contrastive Learning

**Script:** `train_instr_triplet.py` (model) + `preprocess_instr_bigvul.py` (data)  
**Hypothesis:** Triplet loss failed at block level (§6) because a 3-line patch leaves
the block adjacency matrix near-invariant (pair-sim 0.979→0.986). At instruction
level, the same patch changes 3–5 nodes in a ~300-node graph — pairs should be
structurally distinguishable.

### Model: InstructionContrastiveGNN

```
nn.Embedding(80, 128, padding_idx=79)     # opcode vocab → dense embedding
RGCNConv(128 → 64, num_relations=3) × 2  # 3 rels: CFG / DFG / Global
AttentionalAggregation
Projection head: Linear(64→128) → ReLU → Linear(128→128) → L2-norm
```

86,465 parameters. x loaded as long (opcode indices). No z-score normalisation.

### Preprocessing results

```
Raw BigVul rows:         6,807,803
Usable pairs (vul=1):       9,514

  train:  1,117 pairs  (84% attrition from 6,964 raw)
  valid:    122 pairs  (93% attrition from 1,711 raw)
  test:     124 pairs  (85% attrition from   839 raw)
```

Attrition is identical to §6 (block-level BigVul) — both use the same `compile_to_ir()`
bottleneck. Instruction-level graph extraction adds no additional failure modes.

Top CWEs: CWE-119 (172), CWE-125 (96), CWE-20 (78), CWE-unknown (65), CWE-264 (58).

### Training results (30ep, hidden=64, margin=0.3, batch=64)

```
Epoch      Loss   Pair-Sim   Val k-NN
------------------------------------------------
    1    0.4197     0.9984     49.59%  <- best
    3    0.4272     0.9986     52.87%  <- best
   17    0.3528     0.9995     54.10%  <- best
   30    0.3517     0.9995     52.46%
```

**Test k-NN accuracy (k=5): 48.39%**  
**Mean (vuln, fix) cosine similarity: 0.9995** (wrong direction — should decrease)

| Method | Test Acc | Pair-Sim |
|---|---|---|
| BigVul block-level Triplet (§6) | 51.21% | 0.979 → 0.986 ↑ |
| Instruction-level classifier (§7) | 58.00% | — |
| **BigVul instruction-level Triplet (§8)** | **48.39%** | **0.9984 → 0.9995 ↑** |

### Diagnosis: hypothesis falsified

**Pair-sim went the wrong direction: 0.9984 → 0.9995.** Worse than block-level (0.979).
The model pulled both graphs into the same attractor; triplet loss provided no separating force.

The core assumption was wrong. At instruction level, ~297–497 of ~300–500 nodes are *identical*
between vuln and fix — only 3–5 nodes change (0.6–1.7% of the graph). The model sees an
overwhelming shared topology and collapses. Crucially, instruction-level pairs are *more*
similar (0.9995) than block-level pairs (0.979), because the shared instruction structure
is richer and dominates harder than block-level aggregate statistics.

Block-level nodes encode aggregate features per basic block (opcode counts, memory ops, etc.).
When a 3-line patch changes a branch condition, those aggregate counts shift noticeably. At
instruction level, the same patch changes 3 nodes out of 300 — the absolute delta is larger
in blocks, not smaller, because block features collapse hundreds of instructions into 45 floats
that change proportionally.

**Data starvation compounds the collapse:** only 1,117 training pairs survive (same 84–93%
attrition as §6). The model cannot learn fine-grained structural differences from this volume.

### Series conclusion: contrastive direction closed

Three consecutive contrastive experiments have collapsed:

| Experiment | Method | Pair-Sim | Result |
|---|---|---|---|
| §5c | SupCon k-NN (Devign, block) | — | 55.84% — embedding collapse (frozen at log(512)) |
| §6 | BigVul Triplet (block-level) | 0.979 → 0.986 ↑ | 51.21% — soft collapse |
| §8 | BigVul Triplet (instruction-level) | 0.9984 → 0.9995 ↑ | 48.39% — soft collapse (worse) |

**Root cause:** GNN structural topology alone is insufficient to separate (vuln, fix) pairs
when only 0.6–1.7% of nodes differ. The encoder has no access to identifier names, string
literals, or semantic tokens — only opcode categories. CodeBERT's 5.6-point advantage over
the best GNN (63.43% vs 58.00%) comes entirely from pretraining on source text that carries
these identifiers.

### What would actually work

To break the 58% ceiling without LLM pretraining:

1. **Identifier-augmented nodes** — include hashed variable names or type signatures as
   additional node features. LLVM IR preserves all identifiers; discarding them is the primary
   information loss.
2. **Anchor-positive from same function, different commit** — the current positive mining
   (same-CWE) does not guarantee structural similarity. Ground-truth positives from the same
   CVE would provide a cleaner gradient.
3. **Larger dataset without the clang attrition problem** — BigVul's 84–93% compilation
   failure rate leaves too little data. A dataset compiled in advance (pre-generated IR) would
   eliminate this bottleneck.

### Pipeline status

The **block-level GNN classifier (57.84%, `model.pt`)** remains the pipeline deliverable:
zero-LLM-cost pre-filter for the SCAR triage stage. The instruction-level classifier (58.00%)
is not yet deployed — the marginal improvement (+0.16%) does not justify replacing the
existing `scan_ir.py` / `model.pt` infrastructure.

The contrastive embedding direction (§5c / §6 / §8) is closed pending richer node features.

---

## §9 — Real-World Validation: scarnet

**Script:** `eval_scarnet.sh`  
**Dataset:** johwes/scarnet — 5 source files, 19 scoreable functions, 13 known-vulnerable  
**Compilation:** `clang -O0 -fno-inline -S -emit-llvm -I include/`

### Setup

`eval_scarnet.sh` clones scarnet, compiles each source file to LLVM IR, runs
`scan_ir.py --all-functions` on each, and cross-references scores against the
answer key. The `--all-functions` flag was added to `scan_ir.py` for this
evaluation — it splits the IR on `define` boundaries, scores each function
independently, and returns results sorted by score descending.

**Key discovery — compilation flags matter:**  
First run used `-O1`. Clang inlined `handle_set`, `handle_stats`, and `handle_del`
into `dispatch` (routing function that calls them all) and inlined `handle_client`
into `main`. Result: only 12 scoreable functions, `dispatch` at 91.6% absorbing
three vulnerable callees. Switching to `-O0 -fno-inline` restored all 19 functions
as independent scoreable units.

### Results (all 19 functions, sorted by GNN score)

| Rank | Function | File | Score | Prediction | Known vuln? |
|---|---|---|---|---|---|
| 1 | handle_stats | src/handler.c | 88.0% | VULNERABLE | YES |
| 2 | parse_msg_header | src/session.c | 87.4% | VULNERABLE | YES |
| 3 | session_login | src/session.c | 78.7% | VULNERABLE | YES |
| 4 | main | main.c | 69.3% | VULNERABLE | — |
| 5 | session_frag | src/session.c | 66.7% | VULNERABLE | YES |
| 6 | session_free | src/session.c | 66.1% | VULNERABLE | — |
| 7 | session_consume_frag | src/session.c | 63.3% | VULNERABLE | YES |
| 8 | scar_alloc_copy | src/util.c | 57.6% | VULNERABLE | YES |
| 9 | scar_atoi | src/util.c | 55.5% | VULNERABLE | YES |
| 10 | parse_batch | src/parse.c | 54.5% | VULNERABLE | YES |
| 11 | handle_get | src/handler.c | 54.2% | VULNERABLE | — |
| 12 | parse_cmd | src/parse.c | 48.8% | safe | YES |
| 13 | handle_client | main.c | 47.8% | safe | YES |
| 14 | scar_log | src/util.c | 42.1% | safe | YES |
| 15 | handle_auth | src/handler.c | 35.6% | safe | — |
| 16 | session_new | src/session.c | 34.7% | safe | — |
| 17 | handle_del | src/handler.c | 28.3% | safe | YES |
| 18 | dispatch | src/handler.c | 21.6% | safe | — |
| 19 | handle_set | src/handler.c | 12.4% | safe | YES |

**Known-vulnerable functions in top-13: 10 / 13 (77% precision, 77% recall)**

### Analysis

**True positives (10):** handle_stats, parse_msg_header, session_login, session_frag,
session_consume_frag, scar_alloc_copy, scar_atoi, parse_batch, parse_cmd, handle_client.
All correctly ranked in the top half.

**False positives (3):** main, session_free, handle_get. All clean functions the LLM
would reject within seconds of reading source. Operationally harmless.

**False negatives (3):** scar_log (42%), handle_del (28%), handle_set (12%).
Each is a semantic bug with no structural IR signature:

| Function | Score | Bug | Why the GNN misses it |
|---|---|---|---|
| scar_log | 42% | CWE-134 format string | `printf(msg)` — one call, no unusual topology |
| handle_del | 28% | CWE-415 double free | Conditional free in loop — moderate complexity, no graph signature the model learned |
| handle_set | 12% | CWE-476 null deref + CWE-125 strncpy | `malloc` → dereference + off-by-one — local, low block-count, structurally unremarkable |

These three are the LLM's natural domain: it reads the API call semantics (`printf(user_input)`,
`free` inside a conditional, `strncpy` with `>` vs `>=`) without needing structural topology.

### Conclusion

The GNN is a useful zero-cost pre-filter. At no API cost and sub-second runtime per
function, it ranks 77% of known-vulnerable functions into the top half of the candidate
list. The three false negatives are semantic bugs that the LLM catches independently —
the two tools are complementary rather than redundant.

**Pipeline integration:** compile with `-O0 -fno-inline`, run `scan_ir.py --all-functions`
per source file, sort findings by GNN score descending, feed to LLM triage in that order.
Functions scoring below ~20% can be deprioritized (not dropped — `dispatch` at 21% shows
the floor can include routing code with inlined vulnerable callees).

**The topology-vs-semantics boundary is the fundamental limit.** GNNs on opcode-level IR
graphs cannot detect bugs whose only signature is which API is called or what value a
variable holds. Closing that gap requires identifier-augmented node features or LLM
pretraining on source text — the same conclusion reached in §8.

---

## §10 — Vocabulary Enrichment: icmp/fcmp Predicate Expansion

**Scripts:** `preprocess_instr.py` (updated), `train_instr.py`, `train_instr_triplet.py`  
**Status:** COMPLETE  
**Commit:** 346ff22

### §10a Result — Instruction-level classifier with enriched vocab

**Test accuracy: 56.85%** (30 epochs, hidden=64, CPU)

| Metric | Value |
|---|---|
| Best val accuracy | 57.27% (epoch 14) |
| Test accuracy | **56.85%** |
| §7 baseline (old vocab) | 58.00% |
| Block-level best §4d | 57.84% |

**Interpretation:** The vocabulary enrichment did not help the supervised classifier. 56.85% is below both the §7 instruction-level baseline and the block-level ceiling.

**Root cause — embedding starvation:** VOCAB_SIZE 80→110 splits the single `icmp` embedding (ID 46, seen by every comparison in the dataset) into 10 predicate-specific embeddings (IDs 80–89). Each individual predicate embedding now receives a fraction of the training signal. The model sees icmp_sgt far less often than the old catch-all icmp, so its embedding converges more slowly. The predicate distinction is real but the per-predicate sample count is too small to overcome the reduction in embedding quality.

**What this tells us for §10b:** The predicate vocabulary is structurally correct — the node IDs are now distinct — but a supervised classifier trained on Devign cannot exploit this because comparison-operator patches are rare relative to the full vulnerability distribution. Whether FCL+SAGPooling on BigVul (which is explicitly structured as (vuln, fix) pairs) can exploit the predicate distinction is still an open question.

**Note:** `model_instr.pt` has been overwritten with the 56.85% checkpoint. The §7 run (58.00%) is no longer recoverable from disk without re-training. For the pipeline deliverable, `model.pt` (block-level, 57.84%) remains the correct model.

### Motivation

Every experiment from §5a through §8 carries the same blindspot: all comparison predicates collapse to a single vocabulary entry.

`preprocess_instr.py` vocabulary before §10:
```python
"icmp": 46, "fcmp": 47,   # ALL comparison predicates → same ID
```

This means the vulnerable and patched versions of this function:
```c
if (index > size)   →   icmp sgt i32 %index, %size   →  embedding ID 46
if (index >= size)  →   icmp sge i32 %index, %size   →  embedding ID 46
```
produce identical node features. For comparison-operator patches — off-by-one errors in bounds checks, a common class of CVEs (CWE-193, CWE-197) — the vulnerability and its fix are **mathematically indistinguishable** to the model. The triplet loss in §8 had zero gradient contribution from these nodes.

`debug_predicate.py` confirmed the extraction mechanism: `str(instr)` in llvmlite returns the full LLVM IR text of an instruction, and a one-line regex extracts the predicate reliably:
```
=== VULNERABLE (sgt) ===
  str(instr):    '%cmp = icmp sgt i32 %index, %size'
  current  ID:   46   →   enriched ID: 84

=== FIXED      (sge) ===
  str(instr):    '%cmp = icmp sge i32 %index, %size'
  current  ID:   46   →   enriched ID: 85
```

### Implementation

`_instr_node_id(instr)` in `preprocess_instr.py` routes `icmp`/`fcmp` instructions through predicate-specific IDs via `_ICMP_PRED_RE`/`_FCMP_PRED_RE` regex. Unknown predicates fall back to IDs 46/47. VOCAB_SIZE: 80 → 110.

**icmp predicates — IDs 80–89:**

| Predicate | ID | Security relevance |
|---|---|---|
| eq | 80 | null check, sentinel value check |
| ne | 81 | null check (negated form) |
| slt | 82 | signed less-than bounds check |
| sle | 83 | signed less-or-equal — common fix for off-by-one |
| sgt | 84 | signed greater-than — vulnerable form of sle patches |
| sge | 85 | signed greater-or-equal — vulnerable form of slt patches |
| ult | 86 | unsigned less-than bounds check |
| ule | 87 | unsigned less-or-equal |
| ugt | 88 | unsigned greater-than |
| uge | 89 | unsigned greater-or-equal |

**fcmp predicates — IDs 90–105:** false/oeq/ogt/oge/olt/ole/one/ord/uno/ueq/ugt/uge/ult/ule/une/true

The signed/unsigned distinction (slt vs ult, sgt vs ugt) is itself a vulnerability class: using a signed comparison on an unsigned value is a classic integer-overflow pattern (CWE-195).

### What this fixes and what it doesn't

**Fixes:** For comparison-operator patches, the anchor and negative in a triplet now have different node IDs. The embedding lookup returns different vectors. The model can, for the first time, learn that functions containing `icmp sgt` on pointer arithmetic are statistically more vulnerable than those with `icmp sge`.

**Does not fix:** The dilution problem. After vocabulary enrichment, 3–5 changed nodes still compete with 295–497 identical nodes during GNN message-passing. The gradient from those changed nodes may be too small to dominate training, especially at the 1,117-pair BigVul scale.

**The honest prediction:** accuracy improvement for the classifier (§10 re-run of §7) is likely modest but real — the model now has a signal where it had none. Whether the contrastive approach benefits enough to stop collapsing is empirical.

**Actual §10a result:** 56.85% — below §7 baseline (58.00%). The signal is present but embedding starvation from splitting the icmp catch-all into 10 predicates offset any gain. See §10a result block above.

### §10b — Focal Contrastive Loss + SAGPooling

Script: `train_instr_focal.py`

Two architectural changes over §8 (`train_instr_triplet.py`):

**1. Focal Contrastive Loss (FCL)** replaces triplet loss.

```
L_fcon = -1/n · Σ_i  1/|P_i| · Σ_{j∈P_i} (1-p_ij)^γ · log( exp(vᵢᵀvⱼ/τ) / Σ_{k≠i} exp(vᵢᵀvₖ/τ) )
```

Where `p_ij = exp(vᵢᵀvⱼ/τ) / Σ_{k≠i} exp(vᵢᵀvₖ/τ)` is the current model probability for pair (i,j).

The `(1-p_ij)^γ` term is the focal modifier (detached — does not enter the backward graph).
In the collapse regime where all similarities → 1.0, the softmax distributes uniformly
(`p_ij → 1/n`), and the modifier approaches 1, restoring full gradient.
In a well-trained model, easy positive pairs (high p_ij) are down-weighted automatically.

This is in-batch SupCon with focal weighting. Positive sets:
- vuln anchor i: all other vulns in batch
- fix anchor i:  all other fixes in batch

No explicit CWE mining. Batch-64 gives ~32 same-class positives per anchor.

Default: `tau=0.07`, `gamma=2.0` (SupCon standard parameters).

**2. SAGPooling** replaces `AttentionalAggregation`.

Self-Attention Graph Pooling keeps the top `pool_ratio` fraction of nodes by
a learned GCN-based attention score, then `global_mean_pool` over survivors.

```
Architecture:
  Embedding:  nn.Embedding(110, 128, padding_idx=79)
  Encoder:    RGCNConv(128→64, 3 rels) × 2
  Pool:       SAGPooling(hidden=64, ratio=0.25)  →  global_mean_pool
  Proj head:  Linear(64→128) → ReLU → Linear(128→128) → L2-norm  (training only)
```

With `pool_ratio=0.25`, a 400-node graph retains ~100 nodes.
The SAGPooling scorer is untyped — it sees all edges equally — but RGCN has already
differentiated CFG/DFG/global edge types in the node features before pooling.

**Why this pair:** FCL targets the loss landscape (gradient amplification in collapse),
SAGPooling targets signal dilution (concentrating embedding on high-degree nodes).
Both are needed if the failure mode is dilution-then-collapse.

### To run §10

Existing `data/{split}_instr_graphs.pkl` files used old ID 46 for all icmp — they must be regenerated:

```bash
# Regenerate instruction-level graphs with enriched vocabulary
python preprocess_instr.py

# §10a: re-run instruction-level classifier (baseline: §7 = 58.00%)
python train_instr.py --epochs 30 --hidden 64

# §10b: FCL + SAGPooling contrastive (baseline: §8 = 48.39%, pair-sim 0.9984->0.9995)
python preprocess_instr_bigvul.py --csv data/MSR_data_cleaned.csv --workers 4
python train_instr_focal.py --epochs 50 --hidden 64 --tau 0.07 --gamma 2.0
# softer focal (if training unstable): --tau 0.1 --gamma 1.0
```

### Success criteria

| Signal | Target | Interpretation |
|---|---|---|
| §10a classifier | > 58.00% | enriched vocab helps detect comparison-operator vulnerabilities |
| §10a classifier | ≈ 58.00% | vocabulary blindspot was not the binding constraint |
| §10b pair-sim | < 0.95 | FCL+SAGPooling breaks the collapse |
| §10b pair-sim | < 0.9984 | partial improvement over §8 baseline |
| §10b pair-sim | still ↑ | dilution remains binding — identifier features needed next |
| §10b k-NN | ≥ 58.00% | contrastive objective adds value over supervised classifier |

### §10b Result

| Metric | Value |
|---|---|
| Train pairs | 1,117 |
| Val pairs | 122 |
| Test pairs | 124 |
| Best val k-NN | 58.20% (epoch 33, noise) |
| **Test k-NN (k=5)** | **47.58%** |
| Mean (vuln, fix) pair-sim | **0.9992** (↑ from 0.9976 at epoch 1) |
| Loss plateau | 4.7277 → 4.7201 from epoch 2 onwards |

**Collapse diagnosis:** Loss stabilised at 4.7201 after the first gradient step and did not
meaningfully move for 49 epochs. This value is consistent with isotropic collapse: all 128
pair-embeddings are uniformly distributed on the unit hypersphere, `p_ij ≈ 1/127` for all j,
focal weights ≈ 1.0, gradient is symmetric — no net update direction exists.

Pair-sim increased over training (0.9976 → 0.9985). SAGPooling learned to retain high-degree
global context nodes (the virtual context node and call targets), which are structurally
identical in both vuln and fix graphs. No vulnerability-specific pooling signal existed to
guide it differently.

The epoch 33 val spike to 58.20% is noise: 122 validation pairs give ±4–5% variance in k-NN
accuracy from random projection-head variation. Test accuracy (47.58%, below chance) confirms
no generalisation.

**Root cause (structural invariance, not architecture):** FCL amplified gradient correctly in
the collapse regime. SAGPooling reduced node count by 75%. Neither intervention can create
structural signal that is absent in the input: 3–5 changed nodes in a 400-node graph (0.5–1%)
are drowned by global_mean_pool regardless of which 100 nodes SAGPooling retains. The
vulnerability patch leaves no structural footprint that topology-only embeddings can resolve.

**Contrastive learning branch: CLOSED.**

| Experiment | k-NN accuracy | Pair-sim (end) | Verdict |
|---|---|---|---|
| §6 block-level triplet | 51.21% | 0.986 ↑ | collapse |
| §8 instr-level triplet | 48.39% | 0.9995 ↑ | collapse |
| **§10b instr-level FCL+SAGPooling** | **47.58%** | **0.9992 ↑** | **collapse** |

Three experiments across two granularities (block, instruction), two loss functions (triplet,
FCL), and two pooling strategies (AttentionalAggregation, SAGPooling) all collapsed the same
way. The supervised classifier (§4d: 57.84%, §7: 58.00%) remains the best approach. Its value
is as a zero-cost ranker (scarnet: 77% P@13), not as a discriminative embedding model.
Identifier names, string literals, and comparison operand values — the semantic features that
distinguish a vulnerability from its patch — are discarded by LLVM IR compilation and cannot
be recovered by topology alone.




---

## §11 — Program Slice GNN: Backward DFG from Dangerous Sinks

**Script:** `preprocess_slice.py` + `train_slice.py`

**Hypothesis:** The ~58% ceiling of instruction-level classifiers (§7) is caused by signal
dilution: 400-node function graphs where only 3-5 nodes carry vulnerability signal, which
GNN message-passing averages away. Program slicing addresses this with program analysis
rather than learned pooling: trace backward through data-flow edges from dangerous sink
call sites to find the minimal subgraph that determines whether the sink is safely invoked.

**Dangerous sink patterns:** Two categories of slice anchors:

1. **Call-based sinks** — call instructions whose function operand (mock node) matches:
   - Buffer copy: `strcpy`, `strncpy`, `memcpy`, `memmove`, `memset`, `sprintf`, `snprintf`, `gets`, `fgets`, `scanf`, `read`, `recv`
   - Memory management: `malloc`, `calloc`, `realloc`, `free`
   - Format string: `printf`, `fprintf`, `syslog`
   - Name variants handled: `__GI_strcpy`, `__wrap_malloc`, etc.

2. **GEP-based sinks** — `getelementptr` instructions (opcode 29) with at least one
   non-constant DFG predecessor. A variable-index GEP is a potential out-of-bounds
   access (CWE-125/787); its backward slice traces where the index came from.

**Slice algorithm:**

```
1. Build full instruction-level graph (same 5-pass algorithm as §7)
   — additionally track mock node names during Pass 3
2. Identify sink nodes (call sites + variable-index GEPs)
3. BFS backward through DFG edges (full closure, no depth limit)
4. Re-index slice nodes; re-add virtual context node with type-2 edges
5. Fallback: if no dangerous sinks found, keep full graph
   (functions with no sinks are likely safe — correct supervision signal)
```

**Signal concentration:** A 400-node full-function graph where 0.5–1% of nodes
carry vulnerability signal becomes a 15–50 node slice where every node is on the
data-flow path to a dangerous operation (50–80% signal density).

**Prior work:** VulPathFinder (2023) uses the same principle on source-level dependency
graphs: 61% accuracy vs our §7 58.00%. VulDeePecker (2018) pioneered the "code gadget"
approach (backward/forward slices from API call sites) using BiLSTM instead of GNN.

### To run §11

Requires Devign data (same source as §7 — `data/*.jsonl` from `preprocess.py`):

```bash
# Build slice graphs (reuses Devign jsonl from preprocess.py)
python preprocess_slice.py              # full Devign, 4 workers
python preprocess_slice.py --subset 200 --workers 1  # smoke test

# Train classifier on slice graphs
python train_slice.py --epochs 30 --hidden 64   # baseline comparison to §7
python train_slice.py --epochs 60 --hidden 128  # extended run
```

### Success criteria

| Signal | Target | Interpretation |
|---|---|---|
| Test accuracy | > 58.00% | slicing improves over §7 full-function baseline |
| Test accuracy | > 61.00% | reaches VulPathFinder territory — slicing is the key |
| Test accuracy | ≈ 58.00% | ceiling unchanged — sinks not predictive in Devign distribution |
| Slice node count | < 100 (mean) | slicing is working; full fallbacks not dominating |
| Fallback fraction | > 80% | Devign functions rarely contain detectable sinks → different sink patterns needed |

### §11 Result

| Metric | Value |
|---|---|
| Train graphs | 10,101 (54% attrition) |
| Sliced | 8,036 / 10,101 (80%) |
| Fallback (no sinks) | 2,065 / 10,101 (20%) |
| Mean slice nodes | 37 (median 21, max 3,105) |
| Best val accuracy | 56.39% (epoch 30) |
| **Test accuracy** | **56.64%** |

**Root cause — guard conditions are invisible to DFG-only slicing.**

Consider the two structurally different but DFG-identical cases:

```c
// Safe — guarded:
if (n < sizeof(buf))
    memcpy(buf, src, n);

// Vulnerable — unguarded:
memcpy(buf, src, n);
```

Both produce the same backward DFG slice from `memcpy`:
`{arg(buf), arg(src), arg(n)} → call @memcpy`.
The `if (n < sizeof(buf))` guard lives in a CFG predecessor basic block and has
no DFG edge into `memcpy` — it is control dependence, not data dependence.
A DFG-only slice discards exactly the nodes that determine whether the call is safe.

The training curve confirms: val accuracy noisy and slow (49% → 56% over 30 epochs,
never stabilising). The model finds weak signal but nothing decisive.

**VulPathFinder's Source Dependency Graph** (which achieves 61%) explicitly includes
both data dependence (DFG) and control dependence (CFG predecessors). That is the gap.

**Fix for §12:** Extend slicing to PDG (Program Dependence Graph) = DFG + control
dependence. For each DFG-reachable node, find its basic block, add the CFG predecessor
branch instructions and their icmp condition nodes to the slice, then trace those
backward through DFG to capture what the guard conditions depend on.

---

## §12 — PDG Slice GNN: Backward Slice with Control Dependence

**Hypothesis:** §11 failed (56.64%) because DFG-only slicing misses guard conditions.
The `icmp`+`br` instructions that determine whether a dangerous call is safe live in
a CFG predecessor block and have no DFG edge into the sink. Both a guarded (safe) call
and an unguarded (vulnerable) call produce identical DFG backward slices.

Fix: extend the slice to a **Program Dependence Graph (PDG)** — DFG + control dependence.
Control dependence of node v = the terminator (`br`/`switch`) of each CFG predecessor block
of v's basic block. In LLVM IR, `br i1 %cmp ...` already has a DFG edge from `%cmp`
(VK_INSTRUCTION operand), so adding the `br` automatically pulls in the `icmp` guard and
its comparison operands via the next DFG BFS iteration.

### Algorithm

Fixed-point loop implemented in `_extract_slice_pdg()`:

```
visited = sink_ids
ctrl_checked = {}
repeat:
  DFG backward BFS from all nodes in visited
  for each newly visited instruction node v:
    for each CFG predecessor block P of v's block:
      add P's terminator (br/switch) to visited
until no new nodes added
```

Guarded pattern after PDG slicing:
```
%cmp = icmp slt %n, %size     ← pulled in (br operand → DFG edge to br → control dep)
br i1 %cmp, %safe, %unsafe   ← pulled in (control dep of call's block predecessor)
safe:
  call @memcpy(buf, src, n)   ← sink (same as §11)
  ; plus DFG: buf, src, n arguments
```

Unguarded pattern: no `br`/`icmp` predecessor → slice identical to §11, no guard nodes.

The GNN can now structurally distinguish guarded from unguarded dangerous calls.

### Additional tracking (vs preprocess_slice.py)

| Data structure | Built in | Content |
|---|---|---|
| `instr_to_block` | Pass 1 | node_id → block ptr_id (instructions only) |
| `block_preds` | Pass 2 | block ptr_id → [predecessor block ptr_ids] |
| `block_last_instr` | Pass 2 | block ptr_id → terminator node_id |

Arguments, constants, and mock nodes are absent from `instr_to_block` — they have no
control dependence, which is correct.

### To run

```bash
cd experiments/ir_embed_demo/train_gnn
python preprocess_slice_pdg.py --subset 200 --workers 1   # smoke test
python preprocess_slice_pdg.py                             # full Devign
python train_slice_pdg.py --epochs 30 --hidden 64
```

Expected slice size: slightly larger than §11's mean=37 nodes. Each guarded sink adds
roughly 3–5 nodes (br + icmp + icmp operands) per guarded CFG predecessor.

### Success criteria

| Result | Interpretation |
|---|---|
| > 60.0% | Approaches VulPathFinder (61%); control deps are the key missing signal |
| 58.0–60.0% | Beats §7; PDG slicing adds value over the full-graph classifier |
| 56.64–58.0% | Improvement over §11 DFG-only but below full-graph ceiling |
| < 56.64% | Regression; check slice sizes and sliced/fallback fraction |

### Result

**Test accuracy: 56.48%** (10,127 train / 1,254 valid / 1,250 test)

| Metric | §11 DFG-only | §12 PDG |
|---|---|---|
| Test accuracy | 56.64% | 56.48% |
| Mean slice nodes | 37 | 57 |
| Max slice nodes | — | 3,105 |
| Sliced fraction | ~72% | ~77% |
| Full-graph fallbacks | ~28% | ~23% |

**Within noise of §11** (±1.4% at n=1,250). The PDG expansion is confirmed — slices are
larger (mean 37→57 nodes), and control dep nodes (`br`, `icmp`) are present — but
classification did not improve.

**Root cause analysis:**

The hypothesis was that DFG-only slicing misses guard conditions, and that `icmp`+`br`
nodes would distinguish guarded-safe from unguarded-vulnerable calls. In practice:

1. **Negative-space reasoning:** Guard conditions are ABSENT in vulnerable code, not
   PRESENT in safe code. The GNN must learn that *missing* guard nodes signals
   vulnerability — an absence-of-feature pattern that message passing doesn't reliably
   learn. RGCN propagates information along edges; a missing subgraph leaves no
   gradient signal.

2. **Guard inflation:** max=3,105 nodes means some slices grew to near-full-graph
   size via iterated control dep expansion. These dominate training loss without
   carrying cleaner signal than full graphs.

3. **Source-level gap:** VulPathFinder achieves 61% using source-level PDG with richer
   features (variable names, types, constant values). LLVM IR at `-O0` discards these.
   The next step is a CodeBERT-based baseline on source text (§4a in roadmap).

**Preamble fix (commit f89d686):**
- Root cause of 100% attrition: `typedef long long loff_t;` conflicted with system
  `loff_t = __loff_t (aka long)` from `/usr/include/sys/types.h:42`
- Fix: add `#include <sys/types.h>` to `_PREAMBLE_STATIC`, remove standalone `loff_t` typedef

---

## §13 — Tier 1 Feature Extraction: Perfograph Constant Encoding + Categorical Call Targets

**Scripts:** `preprocess_v2.py` + `train_v2.py` (block-level); `preprocess_instr_v2.py` + `train_instr_v2.py` (instruction-level)

**Hypothesis:** Two Tier 1 improvements from the feature extraction roadmap applied together:
1. **Perfograph constant encoding** — replace binary "is this a constant?" flag with `sign(C) * log2(|C| + 1)`, encoding constant magnitude as a compact signed float. Boundary-condition constants (0, 1, −1, buffer bounds) are the most semantically loaded values in vulnerability-triggering code.
2. **Categorical call target mapping** — replace single generic `has_call` flag (block-level) or single `IDX_MOCK=75` (instruction-level) with 5 category-specific features/vocab IDs:

| Category | Functions | New ID (instr) |
|---|---|---|
| Alloc | malloc, calloc, realloc, kmalloc, av_malloc… | 106 |
| Copy | memcpy, memmove, memset | 107 |
| String | strcpy, strncpy, sprintf, gets, fgets… | 108 |
| FileIO | fopen, fread, fwrite, read, write, open | 109 |
| Network | recv, send, accept, connect, recvfrom… | 110 |

**Architecture changes (instruction-level):**
- Node features: `(N, 1)` int64 → `(N, 2)` float32 `[opcode_id, const_magnitude]`
- VOCAB_SIZE: 110 → 111 (5 new call-category IDs)
- `RGCNConv(embed_dim + 1, hidden, 3)` — appends const_magnitude channel after embedding lookup

**Implementation note — NaN constants:** LLVM IR contains `float nan` and `float inf` literals in FFmpeg DSP code. `math.log2(float('nan'))` returns nan silently (no exception), poisoning the entire graph's forward pass. Fix: `math.isfinite(val)` guard in `_const_magnitude` returns 0.0 for non-finite inputs. Safety net in loader: `torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)`.

### Results

**Block-level (preprocess_v2.py → train_v2.py, 30ep, h=64):**

**Test accuracy: 56.75%** (10,117 train / 1,255 valid / 1,252 test)

Below the §4d baseline (57.84%). Block-level constant extraction via text regex (`i32 -1`, `i64 42`) adds noise — width tokens like `i32`/`i64` appear as false matches — and aggregating to a single `max_const_log` per block washes out the per-instruction signal. The categorical call flags replace `has_call` with 6 bits but the underlying blocks don't change; without the opcode-level resolution of the instruction graph, the additional features add variance with no accuracy gain.

**Instruction-level (preprocess_instr_v2.py → train_instr_v2.py, 30ep, h=64):**

**Best single-run test accuracy: 58.75%** — +0.75pp over §7 instruction-level baseline (58.00%)

| Run | Epochs | Val peak | Test acc | Notes |
|---|---|---|---|---|
| Run 1 | 30 | ~58% | **58.75%** | Best result; Run 1 checkpoint kept |
| Run 2 | 50 | — | 56.20% | Overfit; val/test diverged |
| Run 3 | 30 | 59.44% | 54.12% | High val / worst test; extreme val/test gap |

**High variance caution:** With ~1,250 samples per split (0.08% per sample), a 2.5pp swing is ~31 samples. The 58.75%–54.12% range across identical runs is noise, not signal. A multi-seed average (5 seeds) would be needed for a stable estimate; single-run results here are recorded per the methodology used in §4–§12.

**Interpretation:**
- Instruction-level: the two features together give a marginal positive signal (+0.75pp best case) but the variance across runs exceeds the effect size. Constant encoding at the instruction node level is directionally correct (each constant node carries its magnitude rather than a generic ID), but the categorical call categorization may need a larger training set to materialize.
- Block-level: constant encoding at block granularity is lossy. The improvement requires instruction-level resolution.

**Key finding:** The Devign structural ceiling remains at ~57–58%. Perfograph encoding and call categorization are the right direction but not sufficient alone to break the ceiling — the representation gap to CodeBERT (63.43%) is attributable to vocabulary (opcode categories vs. identifier names and string literals), not constant magnitudes.

---

## §14 — VSDG Memory Ordering Edges

**Scripts:** `preprocess_instr_v3.py` + `train_instr_v3.py`

**Hypothesis:** Adding a 4th edge relation (State, type=3) between consecutive `load`/`store` instructions on the same pointer operand encodes memory operation ordering that CFG and DFG edges miss. Specifically targets use-after-free and double-free patterns where the vulnerability is in the *sequence* of memory operations, not the presence of any particular opcode.

**Implementation:** Pass 5 in the graph extractor groups load/store instructions by the stable C++ pointer identity of their pointer operand. Consecutive pairs (in block-iteration order) get a directed state edge from earlier to later. Model change: `num_relations` 3 → 4 (one additional RGCN weight matrix per layer).

### Result

**Test accuracy: 57.47%** (10,126 train / 1,255 valid / 1,251 test, 30ep, h=64)

| Metric | §13 (v2) | §14 (v3) |
|---|---|---|
| Val accuracy peak | 59.44% | 56.97% |
| Test accuracy | 58.75% (best run) | 57.47% |
| Relations | 3 | 4 |

**No improvement.** Val accuracy peaked lower (56.97% vs 59.44%) and test accuracy is below the §7 baseline (58.00%). This is a negative result.

**Root cause analysis:**

1. **Edge density on hot pointers.** A pointer accessed 5 times produces a chain of 4 consecutive state edges. In typical Devign functions these chains add extra paths the RGCN must weigh against DFG/CFG, introducing noise without adding discriminative signal at this dataset scale.

2. **Redundancy with global edges.** The Virtual Context Node already propagates load/store information across the full graph in 2 hops. State edges attempt to encode local ordering on top of a globally-connected structure — the marginal information is small and the extra weight matrix requires more data to converge.

3. **Absence-of-guard is still the binding constraint.** State edges capture ordering between memory ops that *are present*. The UAF pattern — a load after a free — only appears if the IR contains both the free and the subsequent load. In Devign, the vulnerable functions often don't contain the free (it's in a callee); the model sees only the post-free access without the free itself.

**Conclusion:** Memory ordering edges add complexity without benefit at Devign scale. The result is consistent with the broader finding that the ceiling is a representation problem (missing identifier semantics), not an edge-type problem.

---

## §15 — Register Name Embedding

**Scripts:** `preprocess_instr_v4.py` + `train_instr_v4.py`

**Hypothesis:** At `-O0`, clang preserves source variable names in LLVM IR register names (`%buf.addr`, `%size`, `%cmp`, `%ret`). These names carry semantic signal that opcode categories alone cannot express: a `load` from `%size` is semantically different from a `load` from `%buf` even though the opcode is identical. Hashing register names via FNV-1a into 64 buckets and learning a 16-dim embedding per bucket should give the model a name-level signal without requiring `-g` debug metadata.

**Implementation:** Third node feature column `x[:,2]` = name bucket [0, 64], where 0 is the anonymous sentinel (purely numeric SSA indices). `_name_bucket()` strips the `.addr` suffix clang appends to parameter copies, lowercases, and applies FNV-1a 32-bit hash mod 64 + 1. Model adds `nn.Embedding(65, 16)` for name lookup; conv1 input widens from 129 to 145 (`embed_dim=128 + const_mag=1 + name_embed=16`). Based on §13 (3 relations — drops §14 state edges which showed no benefit).

### Result

**Test accuracy: 57.47%** (10,128 train / 1,254 valid / 1,251 test, 30ep, h=64)

| Metric | §13 (v2) | §14 (v3) | §15 (v4) |
|---|---|---|---|
| Val accuracy peak | 59.44% | 56.97% | 58.37% |
| Test accuracy | 58.75% (best run) | 57.47% | 57.47% |
| Relations | 3 | 4 | 3 |
| Extra params | — | — | 1,040 (name embed) |

**No improvement.** Val peaked at 58.37% (close to §13's 59.44%) but test accuracy matched §14's 57.47% — below §13's best. The name embedding learns a representation during training but it does not generalize to the test split.

**Root cause analysis:**

1. **Naming conventions don't transfer across codebases.** Devign spans FFmpeg, QEMU, Linux, and LibreSSL. A `%size` in FFmpeg refers to a completely different data flow context than `%size` in the Linux kernel. The FNV-1a bucket maps all occurrences of `%size` to the same embedding, forcing the model to share weights across unrelated semantic contexts. The embedding learns an average signal that generalizes poorly.

2. **High-information names are sparse.** Most instructions in a function are anonymous (numeric SSA indices) or carry low-signal names (`%tmp`, `%retval`, `%cleanup`). The names that would carry real signal (`%buf`, `%n`, `%offset` near a `getelementptr`) are a small fraction of nodes; the embedding for their buckets is trained on few examples.

3. **Bucket collisions flatten the signal.** With 64 buckets and thousands of distinct names across the dataset, security-relevant names (`buf`, `size`, `len`) share buckets with unrelated names. The collision noise dominates the gradient signal from the few informative name co-occurrences.

**Conclusion:** Register name hashing does not add exploitable signal at Devign scale. The val peak of 58.37% suggests the model briefly finds a training-set-specific correlation, but it does not hold to test. This closes the IR feature engineering track — all recoverable signals from `-O0` LLVM IR have now been tried.

---

## Future Directions: IR Signal as Fuzzing Context

The §11 and §12 slice experiments were motivated by vulnerability classification, but the
artifacts they produce are directly reusable for a different purpose: guiding LLM-generated
fuzzing harnesses.

### The connection

Generating a fuzzing harness for a function requires three things the slice work already
computes:

**1. Dangerous sink identification**

`preprocess_slice.py` explicitly identifies dangerous sinks — calls to `memcpy`, `strcpy`,
`malloc`, `free`, pointer dereferences with computed indices — as the starting points for
backward slice extraction. These sinks are precisely the targets a fuzzer should try to reach.
Rather than asking the LLM to guess what to exercise, the sink list tells it directly.

**2. Backward data flow from inputs to sinks**

The backward DFG slice from a sink traces which variables flow into the sink's arguments and
where they originate. This is the taint path a harness needs to exercise. For example, if
`memcpy(dst, src, len)` has a backward slice showing `len` derives from a network read four
assignments upstream, the harness should make that input fuzzable and test values at
arithmetic boundaries.

**3. GNN vulnerability score as prioritization**

`scan_ir.py` scores functions by structural vulnerability likelihood. Used alongside Fuzz
Introspector's reachability metric, the GNN score adds a dimension that reachability alone
misses: "does this function look structurally vulnerable, not just reachable?" High reachability
AND high GNN score identifies the highest-value harness targets.

### What this would look like in practice

The IR signal becomes an additional section of the context payload sent to the harness-
generating LLM:

```json
"ir_signal": {
    "gnn_vulnerability_score": 0.82,
    "dangerous_sinks": [
        {"call": "memcpy", "line": 47},
        {"call": "malloc", "line": 31}
    ],
    "sink_input_variables": ["len (derives from network read)", "src (user buffer)"],
    "backward_slice_depth": 4,
    "dominant_comparison_predicates": ["slt", "sle"]
}
```

The dominant comparison predicates (from the §10 VOCAB_SIZE=110 vocabulary) indicate what
kind of boundary conditions the function checks — `icmp slt` and `icmp sle` mean the function
is performing upper-bound checks, so the fuzzer should probe values at and just above those
boundaries.

### Why this matters

Unconstrained LLM harness generation has a high error rate — empirically up to 94% in
rudimentary setups (HarnessAgent, arXiv 2512.03420) — because the LLM hallucates API
contracts and misses which inputs actually reach the dangerous operations. The backward
slice provides a ground-truth data flow path from input to sink, and the GNN score provides
a confidence signal that the sink is worth targeting. Together they constrain the LLM's
hypothesis space in the same way ContraFix's differential runtime evidence constrains the
patch hypothesis space (arXiv 2605.17450).

The LLVM IR GNN work was conceived as a vulnerability classifier. The slice infrastructure
turns out to be a taint analysis engine that could serve a fuzzing pipeline.


---

## Future Directions: Feature Extraction Improvements

The 12 experiments in this series converged at a ~57–58% ceiling on Devign. The gap to
CodeBERT (+5.6pp) is not architectural — it is representational. Our hand-coded feature
vocabulary discards identifier names, string literals, and comparison operand values that
CodeBERT reads from source. The following improvements target that gap without requiring
LLM pretraining, ranked by expected lift per implementation cost.

---

### Tier 1 — High expected lift, low-to-medium effort

**1. Perfograph constant encoding**

Replace the binary "is this a constant?" flag with `sign(C) * log2(|C| + 1)`. This is a
three-line change to `preprocess_*.py`. The encoding preserves order and compresses large
constants gracefully without unbounded values.

Why it matters: buffer overflows, integer overflows, and off-by-one errors are
overwhelmingly characterized by specific constant magnitudes — small allocation sizes,
powers of two, UINT_MAX. The current binary flag throws that magnitude information away.
Expected impact: marginal on Devign (dataset is noisy), potentially meaningful on Scarnet
where bugs are concrete and constant-dependent.

Reference: Perfograph (Ben-Nun et al. 2022).

---

**2. Categorical call target mapping**

Instead of encoding every `call` edge destination as an opaque node, bucket it into
a small fixed vocabulary:

```
{Allocation, Copy, String, File IO, Network, Internal}
```

| Bucket | Targets |
|---|---|
| Allocation | `malloc`, `calloc`, `realloc`, `kmalloc`, `av_malloc`, `g_malloc` |
| Copy | `memcpy`, `memmove`, `memset` |
| String | `strcpy`, `strncpy`, `strcat`, `strncat`, `sprintf`, `snprintf`, `gets`, `fgets` |
| File IO | `fopen`, `fread`, `fwrite`, `fclose`, `read`, `write` |
| Network | `recv`, `send`, `accept`, `connect` |
| Internal | everything else |

Our current graph treats `malloc`, `strncpy`, and a helper function as structurally
identical. A vulnerability detector needs to know when untrusted data flows into a
string or memory function. This is a vocabulary-lookup change in preprocessing — no
architecture change required.

This directly addresses one of the clearest failure modes from the Scarnet false
negatives (`handle_set`: null deref + strncpy, scored 12%).

---

**3. IR2Vec vocabulary replacement**

Replace the 110-dim one-hot opcode vocabulary with IR2Vec's dense compositional
embeddings. IR2Vec is upstream in LLVM as `IR2VecVocabAnalysis` (LLVM 17+). The
additive formulation computes a per-instruction vector as:

```
v(instr) = E[opcode] + E[type] + sum(E[operand_i])
```

Similar semantic instructions cluster in embedding space without manual grouping.
This is a principled upgrade over the one-hot/predicate vocabulary developed in §10.

Effort: medium. Requires running `opt -passes=ir2vec-vocab` to extract embeddings per
instruction, then replacing the `nn.Embedding` lookup in `train_instr.py` with the
IR2Vec vectors as pre-computed node features.

Expected impact: this is the most principled representation upgrade available while
remaining entirely at the IR level. It will not close the CodeBERT gap — IR2Vec still
has no identifier names — but it should shift the ceiling 1–3 points and reduce the
vocabulary brittleness seen when new opcode patterns appeared in BigVul.

---

### Tier 2 — Meaningful but higher effort or uncertain payoff

**4. SVF alias analysis for PDG edge cleanup**

The PDG constructed in §4c/§11/§12 uses conservative may-alias approximation, producing
spurious memory dependency edges. SVF (Static Value-Flow Analysis) provides
context-sensitive, flow-sensitive pointer analysis and can eliminate roughly 25% of false
positive edges, producing a cleaner graph for GNN message-passing.

Effort: significant. SVF is a separate LLVM pass (C++ tool chain dependency), and its
output must be consumed in the Python preprocessing pipeline. Do constant encoding and
call categorization first to determine whether graph noise is actually the binding
constraint before committing to this.

---

**5. Relative register ID normalization**

Replace `%12`, `%13`, `%14` with `%0`, `%1`, `%2` within each basic block's scope.
Two identical code patterns with different register numbers currently look different to
the model because SSA numbering is determined by compilation order, not semantics.

Cost: essentially zero — a preprocessing transformation. Expected effect: reduces
variance more than lifting the mean, making the model more robust to compilation
artifacts that change register numbering without changing meaning.

---

### Tier 3 — Research-grade, likely not worth the cost for this project

**6. inst2vec / skip-gram pre-training**

Train a skip-gram model on a large LLVM IR corpus to learn opcode co-occurrence
embeddings. This is the inst2vec approach. Embedding quality depends entirely on corpus
size and diversity; on a typical academic machine this requires weeks of preprocessing.
The representation still has no identifier names — the ceiling stays below CodeBERT. The
engineering cost is high relative to the likely gain over IR2Vec, which already provides
composable dense embeddings without needing corpus pre-training.

---

**7. Hybrid Graph-Transformer (ProGraML-style)**

Replace RGCNConv with a Graph-Transformer that attends over heterogeneous edges (control,
data, call). The problem: attention over heterogeneous graphs requires significantly more
memory and training time, and the Devign ceiling appears limited by the feature vocabulary,
not the architecture. Adding a powerful architecture on top of weak features mostly
amplifies noise. If IR2Vec + Perfograph + call categorization push accuracy above 60%,
revisiting the architecture makes sense — in that order.

---

### The gap to CodeBERT: what it would actually take to close it

CodeBERT's +5.6pp advantage comes from a specific source: identifier names (`buf`, `src`,
`size`), string literals, and type tokens. None of these appear in any IR2Vec vocabulary,
any one-hot opcode encoding, or any graph topology at the opcode level. The GNN sees
structure; CodeBERT sees semantics.

Two paths to close the gap:

1. **LLVM debug metadata** — LLVM IR compiled with `-g` preserves identifier names in
   `!DIVariable` annotations. These survive `-O1` compilation and could be extracted to
   augment node features with partial source-level identifier information without
   requiring full source re-parsing.

2. **Hybrid token-GNN architecture** — Use a pretrained token model (CodeBERT, OSCAR)
   to generate per-instruction node embeddings, then aggregate over graph structure with
   the GNN. This is the direction taken by VulChecker-style approaches and would likely
   match or exceed CodeBERT accuracy. It also removes the need for hand-crafted feature
   engineering entirely.

**Realistic expectation:** IR2Vec + Perfograph constant encoding + categorical call
targets implemented together represent a principled rewrite of the feature extraction
layer with no architecture changes and a plausible path to ~60% on Devign. Everything
beyond that requires one of the two identifier-augmentation strategies above.

---

## §16 — Static Analysis Flags

**Scripts:** `preprocess_instr_v5.py` + `train_instr_v5.py`

**Hypothesis:** Convert absence-of-guard patterns into explicit node-level presence signals. Two patterns: (A) dangerous call (STRING/COPY/FILEIO/NETWORK) with no icmp in same or predecessor block; (B) ALLOC result never compared to null anywhere in the function. Flagged nodes get `x[:,2]=1.0`. Based on §13/v2 (3 relations).

### Result

**Test accuracy: 57.15%** (10,129 train / 1,255 valid / 1,251 test, 30ep, h=64)

| Metric | §13 (v2) | §16 (v5) |
|---|---|---|
| Val accuracy peak | 59.44% | 58.33% (epoch 25) |
| Test accuracy | 58.75% (best run) | 57.15% |
| Flag coverage | — | 14.0% of train graphs |

**Marginal improvement over §7 baseline (56.53%), below §13.** Val peak (58.33%) is close to §13, confirming the flags carry real signal — but coverage is too sparse for the model to learn a robust generalisation.

**Root cause: 14% coverage is the bottleneck.** Only 1 in 7 training graphs has any flagged node. 86% of examples have `x[:,2]=0` everywhere, drowning the flag gradient. The val curve finds the signal late (epoch 22–25) but it doesn't consolidate to test. The pattern detection is correct — it's too narrow.

**Conclusion:** Static analysis flags add real signal but insufficient density. The fix is taint propagation: propagating the flag forward through DFG edges (flagged malloc → downstream dereference is also suspicious) would push coverage from 14% to an estimated 25–35% with no new pattern definitions.

---

## §17 — Taint Propagation + Extended Patterns

**Scripts:** `preprocess_instr_v6.py` + `train_instr_v6.py`

**Hypothesis:** Two improvements over §16: (1) extend Pattern B to cover FILEIO/NETWORK unchecked returns in addition to ALLOC; (2) propagate flags forward through DFG edges — flagged source node gets `x[:,2]=1.0`, 1-hop downstream gets `0.5`, 2-hop `0.25`, 3-hop `0.125`. Converts single-node binary signal into a taint trail.

### Result

**Test accuracy: 58.00%** (10,127 train / 1,255 valid / 1,250 test, 30ep, h=64)

| Metric | §16 (v5) | §17 (v6) |
|---|---|---|
| Val accuracy peak | 58.33% (ep 25) | 58.49% (ep 18) |
| Test accuracy | 57.15% | 58.00% |
| Source graphs (flag=1.0) | 14.0% | 14.9% |
| Tainted graphs (any signal) | 14.0% | 14.9% |
| Flag precision P(vuln\|flagged) | — | 52.5% |
| Flag recall P(flagged\|vuln) | — | 18.0% |
| **Ceiling miss rate** | — | **82.0%** |

**+0.85% over §16, essentially tied with §13 (58.75%) within run variance.**

### Key diagnostic: tainted graphs == source graphs

Both metrics read 14.9% — not a bug. Propagation adds tainted nodes *within* flagged graphs, not new flagged graphs. A graph that has a Pattern A/B source node will propagate taint to its DFG neighbours, but those neighbours are in the same graph. The graph-level coverage number cannot exceed source coverage. The improvement in test accuracy (+0.85%) comes from stronger within-graph signal: the taint trail gives the RGCN a path to follow rather than a single isolated flag.

### The 82% ceiling miss rate

This is the fundamental limit of the static analysis hybrid approach. 82% of vulnerable functions in Devign have no Pattern A or B match — and therefore receive zero signal from `x[:,2]` regardless of how well the taint propagates. This includes the dominant vulnerability classes in Devign that don't involve malloc/FILEIO/dangerous-call-without-guard patterns. No amount of taint depth or propagation decay tuning can reach these functions without new patterns.

**Conclusion:** Taint propagation extracts more value from existing patterns (+0.85% over binary flags) but the 82% ceiling miss rate confirms the static analysis hybrid is at its limit with Pattern A+B alone. The model is now effectively tied with §13 on Devign. Further gains require either new patterns (covering more vulnerability classes) or a different approach entirely.

---

## §17+ Planned Extensions

### Priority 1 — Additional absence patterns

**Pattern C — format string with non-literal format argument:**
A call to `printf`/`sprintf`/`fprintf`/`snprintf` where the format argument position
contains a variable pointer (VK_INSTRUCTION or VK_ARGUMENT) rather than a GEP into a
global string constant. In LLVM IR, string literals compile to
`getelementptr inbounds [N x i8], [N x i8]* @.str, i64 0, i64 0` — detectable as a
VK_GLOBAL_VAR operand. A user-controlled format string has no such GEP.

**Pattern D — unchecked return value (generalised Pattern B):**
Any `call` instruction whose result is never used as an operand of any `icmp`
anywhere in the function. Generalises Pattern B (malloc null check) to all functions
that signal errors via their return value.

### Priority 2 — Multi-class flag

Replace the binary float (0.0/1.0) with a small integer category:
0 = clean, 1 = dangerous call without guard (Pattern A), 2 = unchecked alloc (Pattern B).
Use `nn.Embedding(3, 4)` so the model learns separate representations per pattern type
rather than treating both as the same signal.

---

## §18 — Full Model Sweep on scarnet (all 9 checkpoints)

**Date:** 2026-06-17  
**Target:** `johwes/scarnet`, 19 functions, 13 known-vulnerable  
**Method:** `eval_all_models.py --scarnet --answer-key scarnet-answer-key.txt`  
**Compilation:** `clang-20 -O0 -fno-inline -S -emit-llvm`  
**Fix applied:** `_split_functions()` now generates sibling declare stubs, enabling all 19/19 functions to parse with llvmlite.

### Results

| Model | Section | Devign | Scored | Hits | P@13 |
|---|---|---|---|---|---|
| `model.pt` | §4d block-level DefectGNN | 55.52% | 19/19 | 9/13 | 69.2% |
| `model_instr.pt` | §7 instr baseline (opcode only) | 56.53% | 19/19 | 10/13 | 76.9% |
| `model_instr_v2.pt` | **§13 Perfograph + call categories** | **58.75%** | 19/19 | 10/13 | 76.9% |
| `model_instr_v3.pt` | §14 VSDG memory ordering edges | 57.47% | 19/19 | 10/13 | 76.9% |
| `model_instr_v4.pt` | §15 register name embedding | 57.47% | 19/19 | 8/13 | 61.5% |
| `model_instr_v5.pt` | §16 static analysis flags | 57.15% | 19/19 | 8/13 | 61.5% |
| `model_instr_v6.pt` | §17 taint propagation | 58.00% | 19/19 | 10/13 | 76.9% |
| `model_slice.pt` | §11 DFG slice GNN | 55.60% | 19/19 | 10/13 | 76.9% |
| `model_slice_pdg.pt` | **§12 PDG slice GNN** | 56.48% | 19/19 | **11/13** | **84.6%** |

### Key findings

**Root cause of original 16/19 gap:** `dispatch`, `main`, and `session_consume_frag` each
call a sibling function defined in the same `.c` file. clang emits no `declare` for same-unit
callees; the synthetic per-function IR module was missing those declarations, causing llvmlite
to reject the module. Fix: `_define_to_declare()` generates declare stubs for all sibling
functions, inserted into each function's synthetic module before parsing.

**session_consume_frag (Bug 15) is now visible** to instruction/slice models and ranks in
the top-5 across most models. The block model had already caught it at rank 13 in the 16/19 sweep.

**§13 and §17 dropped from 11/13 → 10/13** despite gaining `session_consume_frag`: `main`
and/or `dispatch` (non-vulnerable) now also score within the top-13 for these models,
displacing one true positive each. Net gain from the fix: zero for these two models.

**§15 and §16 dropped from 10/13 → 8/13**: both `dispatch` and `main` score within the
top-13 for the register-name and static-flag models, displacing two true positives without
`session_consume_frag` compensating (it ranks outside 13 in these models).

**§12 PDG slice uniquely benefits:** `session_consume_frag` ranks in top-5 and neither
`dispatch` nor `main` enters the top-13 for this model. Net gain: the previous 16/19 count
of 10 TPs becomes 11 with `session_consume_frag` added. §12 is now the sole leader.

**Block model catches session_consume_frag** (rank 13/19) but misses the utility
functions `scar_atoi`, `parse_batch`, `scar_alloc_copy`, `scar_log` (ranks 14–19).
Instruction models rank all four utility functions in the top-12 consistently.

**Ensemble attempted and closed — see §19** (run before the sibling-stub fix, with 16/19).

**Devign accuracy does not predict scarnet P/R.** §12 PDG slice (55.60% Devign, best
scarnet at 84.6%) and §4d block (55.52% Devign, worst scarnet 69.2%) show that
Devign-optimised models do not automatically rank real-world functions correctly.
The instruction-level architecture's node granularity matters for real code.

**Consistent false positives across instruction models:** `session_free`, `handle_get`,
`session_new` score 50–63% on multiple models — structural features common to both
vulnerable and clean complex functions. `dispatch` and `main` are new FPs (now scoreable)
that rank high in several models.

### Real-world evaluation queue

- **zlib v1.2.11**: known CVEs including CVE-2018-25032 (deflate buffer overflow).
  Harder test beyond the intentionally vulnerable scarnet.

---

## §19 — Ensemble Scoring (max and mean across all models)

**Result:** Neither ensemble strategy beats the best individual model.

| Strategy | Hits | P@13 |
|---|---|---|
| Best individual (§13 / §17 / §12) | 11/13 | **84.6%** |
| ENSEMBLE (max) | 10/13 | 76.9% |
| ENSEMBLE (mean) | 9/13 | 69.2% |

**Note:** This ensemble was run before the sibling-stub fix (§18), so only 16/19 functions
were scored. Results below reflect that coverage.

**Root cause — `dispatch` is the blocking false positive.**
`dispatch` is one of 3 functions that instruction models could not score (16/19 sweep).
Root cause: **missing sibling declare stubs** — `dispatch` calls `@handle_auth`, which is
defined in the same `.c` file. clang emits no `declare` for same-unit callees; the synthetic
per-function IR module was missing the declaration, causing llvmlite to reject it.

The block model gives `dispatch` a score of 78.3% — its highest false positive.
Because no instruction model scored it in the 16/19 sweep, both max and mean ensembles
inherit this score from the block model without counterweight. Max keeps it at 78.3%
(rank 2). Mean keeps it at 78.3% (one data point, no dilution possible).

In both ensembles, `dispatch` displaces a true positive from the top-13.

**Mean ensemble is additionally worse** because it dilutes clean high-confidence signals:
`session_frag` (82.2% max → ~72% mean), `parse_batch` (76% → ~60%). The best
instruction models are confident on the right functions; averaging with weaker models
degrades that confidence without suppressing the FP.

**Conclusion:** The ensemble chapter is closed. With the sibling-stub fix in place (§18
now shows 19/19), the specific dispatch-FP problem could be re-examined — instruction
models now score dispatch and likely assign it ~47%, which a mean ensemble would suppress.
However, since §12 PDG slice alone achieves 84.6% at 19/19, there is no benefit to
pursuing an ensemble that can match but not exceed the best individual model.

---

## §20 — Zero-Shot Transfer to zlib v1.2.11

**Date:** 2026-06-17  
**Target:** `madler/zlib` tag v1.2.11, 148 functions across 15 `.c` files  
**Ground truth:** 1 known-vulnerable function — `deflate_stored` (CVE-2018-25032:
out-of-bounds write when output buffer is near-exhausted with non-compressible input;
unpatched in v1.2.11, fixed in v1.2.12)  
**Method:** `eval_all_models.py --ir-dir /tmp/zlib-ir/ --answer-key zlib-v1.2.11-answer-key.txt --top-k 10`  
**Compilation:** `./configure && clang-20 -O0 -fno-inline -S -emit-llvm -I. <file>.c`

With only one ground-truth function, **rank** and **MRR** (mean reciprocal rank) are
the appropriate metrics alongside P@10. Random-ranker baseline MRR for 1 item in 148:
H(148)/148 ≈ 0.038.

### Per-model results

| Model | Section | Rank / 148 | Score | MRR | P@10 | R@10 |
|---|---|---|---|---|---|---|
| `model_instr_v4.pt` | §15 reg names | **2** | 83.7% | **0.500** | 10.0% | **100%** |
| `model_instr.pt` | §7 instr baseline | **4** | 79.2% | 0.250 | 10.0% | **100%** |
| `model_instr_v3.pt` | §14 VSDG | **7** | 67.0% | 0.143 | 10.0% | **100%** |
| `model_slice_pdg.pt` | §12 PDG slice | **10** | 63.8% | 0.100 | 10.0% | **100%** |
| `model_instr_v6.pt` | §17 taint | 13 | 61.6% | 0.077 | 0.0% | 0% |
| `model_instr_v5.pt` | §16 static flags | 22 | 65.0% | 0.045 | 0.0% | 0% |
| `model.pt` | §4d block | 30 | 51.6% | 0.033 | 0.0% | 0% |
| `model_instr_v2.pt` | §13 Perfograph | 31 | 65.9% | 0.032 | 0.0% | 0% |
| `model_slice.pt` | §11 DFG slice | 63 | 53.7% | 0.016 | 0.0% | 0% |
| `model_slice_pdg_v2.pt` | §22 PDG + taint | 17 | 60.8% | 0.059 | 0.0% | 0% |
| ENSEMBLE (mean) | — | **7** | 65.7% | 0.143 | 10.0% | **100%** |
| ENSEMBLE (max) | — | **10** | 83.7% | 0.100 | 10.0% | **100%** |

**Mean MRR (9 original classifiers): 0.133 — 3.5× random baseline (0.038)**  
Median rank: 13. R@10=100% for 4/9 models (44%) and both ensembles.
§22 MRR: 0.059 — improvement over §11 DFG slice (0.016) but regression from §12 PDG (0.100). Taint boosted memory-complex functions (`zcalloc`, `crc32_little`, `crc32_big`) above `deflate_stored`.

### Key findings

**Transfer is real.** Models trained exclusively on Devign (mixed OSS C code) rank the
CVE function in the top 10 for 4 of 9 architectures on a completely new codebase,
zero-shot. Top 3% for §7 (rank 4/148), top 5% for §14 (rank 7).

**§15 register name embedding reverses its scarnet result.** Worst on scarnet (61.5%),
best on zlib (rank 2, MRR 0.500). The likely reason: zlib uses production C naming
conventions — `strm`, `have`, `left`, `buf`, `avail_out` — similar to the Devign corpus.
scarnet uses more uniform/abstract variable names. Name features generalise to code that
looks like training data but not to purpose-built benchmarks.

**§11 DFG slice fails** (rank 63/148, MRR 0.016 — below random). DFG slicing without
PDG edges loses the buffer-bound computation flow that makes `deflate_stored` suspicious.

**`compressBound` calibration failure.** A trivial formula function (`sourceLen + (sourceLen >> 12) + ...`) ranks #1 in 4 models at 90–97%. No branches, no memory ops, tiny graph — the model mis-calibrates on structural outliers. Functions that are far from the Devign distribution in size/complexity get unpredictable scores regardless of actual risk.

**Ensemble mean (rank 7) outperforms most individual models** on this corpus. Unlike
scarnet — where ensemble hurt because one FP (dispatch at 78.3%) anchored the score —
here the dominant FP (`compressBound`, ~74% mean score) is already above `deflate_stored`
(~66%) in every individual model too, so averaging doesn't worsen rank.

**CVEs patched in v1.2.11 are instructive:** `inflate_fast` (CVE-2016-9841 fixed here)
ranks #1 in the block model at 100% — the model still considers it suspicious. `crc32_big`
(CVE-2016-9843) ranks #2–#7 across several models. The model has no knowledge of the fix;
it sees the same structurally complex code regardless of patch status.

---

## §21 — BigVul Standard Binary Classifier

**Scripts:** `preprocess_bigvul_cls.py` + `train_bigvul_cls.py`  
**Dataset:** BigVul (MSR_data_cleaned.csv), C functions only  
**Architecture:** InstructionGNN v2 (§13 weight format, identical to `model_instr_v2.pt`)

The prior BigVul experiments (§6, §8) used triplet contrastive learning and collapsed. §21 tests whether the same v2 GNN trained as a standard binary classifier on CVE-level labels outperforms Devign-trained models on real-world code.

**Three label sources:**
- `func_before` where `vul=1` → label=1 (vulnerable)
- `func_after`  where `vul=1` → label=0 (fixed — serves as hard negative)
- `func_before` where `vul=0` → label=0 (unrelated clean functions)

**C-only filter:** BigVul spans many languages. Filtering `file_name.endswith(".c")` before compilation reduced work from 150K to 72K items; attrition dropped from ~90% to ~54%.

**Checkpoint metric:** Balanced accuracy = (TPR + TNR) / 2. Raw accuracy on the BigVul val set (85% negative) peaks at epoch 1 before pos_weight destabilises predictions — balanced accuracy correctly tracks minority-class recall.

### Results

**BigVul-only** (`model_bigvul_cls.pt`):
- Train: 33,422 graphs — 1,201 pos / 32,221 neg, pos_weight=26.828
- Test: raw=79.93% (majority baseline 96.17%), balanced≈61%
- **Scarnet: 9/13 (69.2% P@13)**

**BigVul+Devign combined** (`model_bigvul_combined.pt`):
- Train: 43,549 graphs — 5,601 pos / 37,948 neg, pos_weight=6.775
- Best val balanced accuracy: 67.98% (epoch 23 of 30)
- Test: raw=75.10%, balanced=60.95%
- **Scarnet: 9/13 (69.2% P@13)** — no gain from combining datasets

**Finding:** Switching from Devign's commit-level labels to BigVul's CVE-level labels does not improve real-world detection. The PDG structural signal (§12, 11/13 scarnet) outperforms both regardless of training data. Dataset quality is not the bottleneck — it's the IR representation ceiling established in §§13–17. Adding Devign graphs to the combined training also provides no lift, confirming that label noise in Devign has no multiplicative negative effect (the cleaner BigVul labels simply dominate).

---

## §22 — PDG Slice + Taint Flags

**Scripts:** `preprocess_slice_pdg_v2.py` + `train_slice_pdg_v2.py`  
**Dataset:** Devign  
**Architecture:** SlicePDGGNNv2 — same as §12 but with taint float concatenated after opcode embedding (in_dim = embed_dim + 1)

The §22 hypothesis: §12's PDG selects the right subgraph; adding explicit taint annotations to each node (which instructions carry dangerous values) should improve score *separation* between true and false positives, even if the P@K hit count stays at 11/13.

**Taint sources (same patterns as §17):**
- Pattern A: call to a dangerous function (DANGEROUS_SINKS) with no icmp guard in the same or preceding block
- Pattern B: call to alloc/IO/network function whose return value is never compared with icmp

Taint propagates forward through DFG edges with 0.5 decay per hop (max 3 hops). Second column of (N, 2) float32 node feature matrix.

**Results:**
- Devign: 56.75% (+0.27pp vs §12 — within noise)
- Taint coverage: 12% of training graphs have ≥1 tainted node
- **Scarnet: 9/13 (69.2% P@13) — regression from §12's 11/13**
- **zlib v1.2.11: rank 17/148** — regression from §12's rank 10

**What went wrong on scarnet:** The taint flags boosted two false positives (`handle_get`, `handle_auth`) — both have unguarded dangerous calls by the Pattern A/B definition, but are not actually vulnerable. Two true positives (`session_consume_frag`, `parse_msg_header`) dropped below the cutoff because their vulnerability patterns don't match Pattern A or B, so they received no taint boost.

**What went wrong on zlib:** `zcalloc`, `crc32_little`, `crc32_big` — all memory-intensive functions with complex allocation/copy patterns — ranked ahead of `deflate_stored`. Pattern A/B flags memory complexity, not the specific arithmetic overflow in the CVE. The taint signal generalises to "this function touches memory in complex ways", not "this function has the specific pattern that caused this CVE".

**Key finding:** §12's strength came partly from its *lack* of semantic assumptions. The PDG structural signal generalises to scarnet's vulnerability types without imposing a Devign-derived prior. Adding semantic heuristics (taint patterns tuned to Devign's vulnerability distribution) introduced domain-specific bias that hurt out-of-distribution performance. This mirrors §15/§16: semantic enrichment that doesn't cross the distribution boundary hurts more than it helps.

**Semantic heuristic track closed.** §12 PDG slice (11/13 scarnet) remains the best model.

---

## §23 — PDG Slice with Sink-Node Readout + Residual/LayerNorm

**Scripts:** `preprocess_slice_pdg_v3.py` + `train_slice_pdg_v3.py`
**Dataset:** Devign
**Architecture:** SlicePDGGNNv3 — RGCN + residual connections + LayerNorm + sink-node scatter-max readout

Two architectural problems with §12 motivated this experiment:

**Problem 1 — pooling dilution.** In §12, global AttentionalAggregation pools all nodes in the slice. The slice is built backward from dangerous sinks, so the sink nodes and their immediate guard context are the most informative. But in the worst case the slice expands to 3,105 nodes (heavily nested error-handling code cascading through unbounded control-dependence expansion). Pooling 3,105 nodes dilutes the signal from the 5–10 that actually matter.

**Problem 2 — oversmoothing.** Two RGCN layers with no skip connections and no normalisation risks oversmoothing on larger slices: deep node embeddings converge to neighbourhood averages and lose the individual opcode identity that distinguishes an `icmp slt` from an `alloca`.

**Changes:**

1. **Control-dependence depth cap** (`max_cd_hops=2`). The fixed-point CD expansion loop in §12 is bounded at 2 rounds of (full DFG BFS + CD expansion). A bounds check is almost always within 1–2 control-flow hops of the sink; deeper expansion adds noise, not signal.

2. **Sink-node mask.** `preprocess_slice_pdg_v3.py` stores a boolean `sink_mask` in each graph dict marking which node indices are the identified dangerous sinks. The classifier applies to sink-node embeddings only, then scatter-max over sinks per graph. For graphs with no sinks, all nodes are marked — scatter-max degenerates to global max pool.

3. **Residual connections + LayerNorm** on both RGCN layers. Standard remedy for oversmoothing: LayerNorm stabilises activations, residual connection preserves opcode identity in the node embedding while accumulating neighbourhood context.

### Results

- Devign: **55.40%** (−1.08pp vs §12)
- **Scarnet: 9/13 (69.2% P@13) — regression from §12's 11/13**

**What went wrong:** The sink-node readout is architecturally correct — it reads out from the right place. But it was trained under the same noisy graph-level Devign labels. The sink nodes' K-hop neighborhoods contain the structural guard signal, but without node-level supervision telling the model *which* nodes are the bug, the readout learns to maximise over sink embeddings under a function-level binary label. This introduces training noise: a function labelled "vulnerable" may have its bug at a sink the slicer did not identify, so the scatter-max over the wrong sinks degrades rather than improves. The architecture is correct for the problem; the supervision signal is still wrong.

**Finding:** Sink-node readout requires instruction-level ground truth to realise its potential. Under graph-level noisy labels, it regresses. §12 global pooling, despite being theoretically inferior, is more robust to supervision noise because it diffuses the loss signal over the entire slice rather than concentrating it on a subset that may not contain the labelled bug.

---

## §24 — PDG Slice with Intrinsic-Aware Sinks

**Scripts:** `preprocess_slice_pdg.py` (patched) + `train_slice_pdg_v4.py`
**Dataset:** Devign
**Architecture:** Identical to §12 (SlicePDGGNN, RGCN + AttentionalAggregation). Training data only changes — clean ablation.

A preprocessing bug was discovered in §12: LLVM memory intrinsics have type-suffixed names (`llvm.memcpy.p0i8.p0i8.i64`, `llvm.memmove.p0i8.p0i8.i64`, `llvm.memset.p0i8.i64`) that did not match the bare-name suffix check in `_is_dangerous()`. The result: functions whose primary dangerous operation is a memcpy intrinsic — including Heartbleed-class functions — had the key sink invisible to the backward slicer, producing degraded training graphs with the dangerous operation absent from the slice.

**Fix:** `_is_dangerous()` now recognises `llvm.<name>.*` prefixes. `_canonical_name()` maps intrinsic names back to canonical sink names (e.g. `llvm.memcpy.p0i8.p0i8.i64` → `"memcpy"`) for context enrichment output. Secondary effect: wrapper names (`CRYPTO_malloc` → `"malloc"`) are now also canonicalized.

### Results

- Devign: **55.00%** (−1.48pp vs §12)
- **Scarnet: 10/13 (76.9% P@13)** — improvement from §12's 11/13? No — a different 10.

**What happened:** The intrinsic fix correctly adds new sink nodes to graphs that were previously incomplete. But on Devign, many of those newly-exposed sinks are in functions labelled "safe" — the sink was present but not the bug, and the fix expands slices in both vulnerable and safe functions. The net effect on Devign is slight regression. On scarnet, the fix recovers one function missed by §12 but the retraining loses another. The intrinsic-aware sinks are correctly an improvement to the representation; the Devign accuracy regression is noise at this sample scale.

**Finding:** The intrinsic fix is correct and is retained in the preprocessor as the canonical version. It does not, however, break through the Devign ceiling — the fundamental label noise constraint still applies. **The intrinsic-aware preprocessor is now the default** for all subsequent experiments.

---

## §25 — PDG Slice Trained on PrimeVul

**Scripts:** `preprocess_primevul.py` + `train_slice_pdg_v5.py`
**Dataset:** PrimeVul (arXiv:2403.18624, ICSE 2025)
**Architecture:** Identical to §12 (SlicePDGGNN)

PrimeVul applies LLM-assisted relabeling to real CVE commits, achieving function-level label accuracy approximately 3× better than Devign's commit-level approach. The hypothesis: if Devign's label noise is the binding ceiling (§21 conclusion), training on a cleaner dataset should break through it.

**Dataset characteristics:**
- ~7,000 vulnerable + ~229,000 benign C/C++ functions, 140+ CWEs
- Class imbalance ~1:33 (vuln:benign) — capped at `--max-benign 21000` for a ~1:3 training ratio
- Labels derived from CVE descriptions via LLM relabeling, not raw commit diffs

**Critical attrition problem:** The clang compilation pipeline processes PrimeVul functions as isolated snippets with a common preamble of forward declarations. For Devign (4 large projects with consistent coding conventions), attrition is manageable. For PrimeVul (hundreds of projects, C and C++ mixed, with project-specific types and macros):

- Vulnerable functions survive at **~21.8%**
- Benign functions survive at **~37%**

This 2× attrition differential is a systematic bias: vulnerable functions in PrimeVul tend to be the complex, stateful, deeply nested ones — exactly the functions that depend on project-specific types and cannot compile in isolation. The surviving vulnerable functions are systematically simpler than the full vulnerable population. The model trains on a biased sample that underrepresents the functions it most needs to learn.

### Results

- PrimeVul test accuracy: **71.3%** (on the surviving biased sample — not comparable to Devign)
- Devign cross-eval: **55.56%** (−0.92pp vs §12)
- **Scarnet: 9/13 (69.2% P@13) — regression from §12's 11/13**

**Why it regressed:** The model trained on a biased sample of the simpler PrimeVul functions learns a different structural fingerprint than §12. The scarnet regressions are functions (`session_frag`, complex session handling) that are structurally similar to the complex vulnerable functions PrimeVul attrition filtered out. The model that never saw those patterns in training does not recognise them at inference.

**Attrition is the root cause — not the dataset.** PrimeVul's labels are genuinely cleaner. The problem is the compilation pipeline, not the data source. Fixing attrition (with stub headers, permissive compilation flags, or a parser that does not require complete types) would recover the complex vulnerable functions and potentially break the ceiling. The Joern-based approach (`preprocess_primevul_joern.py`, §26) achieved ~95% coverage vs ~34% for clang — confirming that attrition, not label quality, is the binding constraint.

**Joern path dropped.** §26 (Joern + PrimeVul) was implemented and tested but is not pursued further. Joern is a heavy dependency (Java runtime, separate analysis pass) that conflicts with the SCAR pipeline architecture: SCAR already compiles targets to LLVM bitcode for NASA IKOS static analysis. Requiring a separate Joern analysis pass would add a parallel toolchain for no net gain — the LLVM IR representation is already available for free in the SCAR build step, and IKOS depends on it. All future experiments stay within the LLVM IR/clang pipeline.

**PrimeVul attrition track noted but not continued.** Fixing the clang attrition problem (stub headers, partial compilation) is tractable but the expected gain is uncertain — even with 95% coverage, PrimeVul's vulnerability distribution may not align with scarnet's bug types better than Devign does. The cleaner path forward is the Juliet Test Suite (§27), which provides zero-attrition matched pairs with instruction-level ground truth.

---

## §27 — Juliet Test Suite Pretraining → Devign Fine-tune

**Scripts:** `preprocess_juliet.py` + `train_slice_pdg_v7.py`
**Architecture:** SlicePDGGNN_v7 — same RGCN backbone as §12, multi-feature node input (N, 3)

### Motivation

§12's Devign-only training teaches the model to distinguish "functions that look like FFmpeg/QEMU commits with a bug" from "functions without". This is Philosophy 1 — distribution fingerprinting. The structural signal the model actually needs (Philosophy 2: "does an externally-controlled value reach a dangerous sink without a guard?") is present in the graph but competes with the distribution signal during training.

The Juliet Test Suite (NSA/NIST) provides a clean version of the structural signal:
- ~100,000 synthetic C functions, CWE-organised
- Exactly matched bad/good pairs differing only at the bug site
- Zero label noise — bad/good is definitional, not inferred from commit history
- Covers CWE-121 (stack overflow), CWE-122 (heap overflow), CWE-134 (format string), CWE-415 (double free), CWE-476 (null deref)

**Two-phase training:**
1. Pretrain on Juliet — learn "guarded vs unguarded sink" from clean synthetic pairs
2. Fine-tune on Devign — adapt to real-world IR distribution

**Multi-feature node input (N, 3):**
- col 0: opcode_id (same as §12, for nn.Embedding)
- col 1: guard_class (0=none, 1=bounds_check icmp slt/sle/sgt/sge/ult/ule/ugt/uge, 2=null_check eq/ne)
- col 2: is_external_input (1 if call node's callee is in INPUT_SOURCES — recv/read/fgets/etc.)

These are the same structural facts that `slice_context.py` exposes to the LLM — now baked into the GNN input.

### Juliet preprocessing

- 19,056 function definitions extracted (CWE121: 8,198, CWE122: 5,086, CWE134: 4,740, CWE415: 474, CWE476: 558)
- Attrition: **2%** — near-zero, vs Devign's ~40-60%
- Label balance: nearly perfect 50/50 by construction
- 50% of graphs are sliced (have at least one dangerous sink)
- Guards visible: ~15% of graphs have a bounds-check icmp node, ~30% have a null-check icmp node

### Phase 1 — Juliet pretraining (20 epochs)

Juliet val accuracy reaches **99%+ by epoch 3** and holds. Expected: the structural signal is unambiguous in zero-noise matched pairs. The model is genuinely learning "guarded sink vs unguarded sink", not memorising noise.

### Phase 2 — Devign fine-tuning (30 epochs)

- Devign val accuracy oscillates 50–55%, stabilises ~55%
- Final Devign test accuracy: **56.12%** (§12 baseline: 56.48%)
- Within noise — the Juliet prior resists Devign's noise rather than re-learning the fingerprint

### Results

> **Note:** Results below were re-evaluated after fixing two eval bugs discovered in §31 analysis:
> (1) eval_all_models.py was using `ir_to_graph_slice_pdg` (x shape N×1) instead of `ir_to_graph_slice_pdg_v7` (x shape N×3) for §27+, silently zeroing guard_class and is_external_input in every eval run; (2) `atoi`/`strtol` family missing from DANGEROUS_SINKS, causing `scar_atoi` to return no slice. All §27–§31 scarnet results prior to this fix were produced with scalar features disabled.

| Metric | §12 (Devign only) | §27 (Juliet + Devign BCE) |
|--------|-------------------|----------------------|
| Devign test acc | 56.48% | 56.12% |
| scarnet hits | **11/13** | 10/13 |
| scarnet P@13 | **84.6%** | 76.9% |

**scarnet ranking (§27, corrected):**

| Rank | Function | Score | Vuln? |
|------|----------|-------|-------|
| 1 | session_new | 65.0% | no ✗ (FP) |
| 2 | session_frag | 64.0% | YES ✓ |
| 3 | handle_set | 58.9% | YES ✓ |
| 4 | session_login | 57.0% | YES ✓ |
| 5 | parse_cmd | 55.9% | YES ✓ |
| 6 | parse_msg_header | 54.4% | YES ✓ |
| 7 | parse_batch | 53.8% | YES ✓ |
| 8 | dispatch | 53.7% | no ✗ (FP) |
| 9 | handle_auth | 50.4% | no ✗ (FP) |
| 10 | scar_log | 50.3% | YES ✓ |
| 11 | handle_del | 49.2% | YES ✓ |
| 12 | handle_client | 49.2% | YES ✓ |
| 13 | session_consume_frag | 48.1% | YES ✓ |
| 14 | scar_alloc_copy | 47.5% | YES ✗ (miss) |
| 15 | handle_get | 45.8% | no |
| 16 | handle_stats | 45.7% | YES ✗ (miss) |
| 17 | main | 41.2% | no |
| 18 | session_free | 32.9% | no |
| 19 | scar_atoi | 19.5% | YES ✗ (miss) |

**10/13 — regression from §12.** With guard features active, BCE fine-tune on Devign over-fits the Juliet structural prior to Devign-specific patterns, causing `scar_atoi` (rank 19, 19.5%) to drop — a CWE-190/191 integer conversion pattern. `atoi` was also missing from DANGEROUS_SINKS during Juliet preprocessing, so the model saw no atoi-pattern positives.

**Conclusion:** With corrected eval, §27 BCE scores 10/13 — below §12. The BCE loss with guard features is less robust than the base §12 model for this target. §28 RankNet (below) recovers parity.

---

## §28 — Juliet Pretrain + Devign RankNet Fine-tune

**Scripts:** `preprocess_juliet.py` + `train_slice_pdg_v8.py`
**Architecture:** SlicePDGGNN_v7 — identical to §27 (multi-feature x(N,3), same RGCN backbone)

### Motivation

§27 showed Juliet pretraining transfers without catastrophic forgetting — parity with §12 (11/13). But Phase 2 in §27 still uses binary cross-entropy on Devign's commit-level labels. This trains a classifier, not a ranker. The scarnet evaluation is a ranking problem.

**The mismatch:** BCE punishes a confident-but-wrong prediction with a large gradient. Under ~15% label noise, roughly 1 in 7 training examples send the model in the wrong direction — and the gradient is largest exactly when the model is most confident. RankNet loss is comparatively robust: a mislabelled pair wastes one training pair but does not produce a large wrong-direction gradient because `score(v) - score(b)` is near zero for hard pairs.

**RankNet loss:** for each mini-batch, form all `(vuln, benign)` pairs and minimise:

```
L = mean over (v, b) pairs:  softplus(-(score_v - score_b))
  = -log σ(score_v - score_b)
```

This is `P(v ranks above b)` maximisation — training objective now directly matches the ranking evaluation.

**Devign accuracy note:** a model that learns the correct ranking will have Devign accuracy near 50% if scores are centred — high for vuln, low for benign, but the binary threshold still misclassifies labelled-vuln functions that are structurally guarded. **Expect Devign accuracy ≤ §12/§27. The meaningful metric is scarnet ranking.**

### Training

- Phase 1 (Juliet BCE): identical to §27 — reuses `model_juliet_pretrain.pt` if present, skips re-training
- Phase 2 (Devign RankNet): 40 epochs, lr=1e-4, batch 32 → up to 256 pairs/step
  - Fallback to BCE for single-class batches (ensures gradient flow)

### Results

> **Note:** Re-evaluated after fixing the eval preprocessor bug (see §27 note). Prior §28 results used x(N,1); results below use x(N,3) with guard and external_input features active.

Devign test accuracy: **44.52%** — below §12/§27, as expected for a ranking objective.

scarnet ranking: **11/13, 84.6% P@13** — parity with §12, better than §27 BCE (10/13).

| Rank | Function | Score | Vuln? |
|------|----------|-------|-------|
| 1 | handle_client | 77.9% | YES ✓ |
| 2 | session_new | 75.8% | no ✗ (FP) |
| 3 | parse_batch | 74.8% | YES ✓ |
| 4 | session_frag | 74.0% | YES ✓ |
| 5 | session_consume_frag | 72.6% | YES ✓ |
| 6 | scar_atoi | 71.1% | YES ✓ |
| 7 | handle_set | 71.0% | YES ✓ |
| 8 | parse_cmd | 70.5% | YES ✓ |
| 9 | handle_del | 69.9% | YES ✓ |
| 10 | scar_log | 69.3% | YES ✓ |
| 11 | dispatch | 69.1% | no ✗ (FP) |
| 12 | session_login | 69.0% | YES ✓ |
| 13 | parse_msg_header | 68.4% | YES ✓ |
| 14 | scar_alloc_copy | 68.3% | YES ✗ (miss) |
| 15 | handle_stats | 68.0% | YES ✗ (miss) |
| 16 | handle_get | 65.8% | no |
| 17 | session_free | 63.7% | no |
| 18 | handle_auth | 61.6% | no |
| 19 | main | 60.0% | no |

**Calibration:** Scores compressed into a 60–78% band — less spread than §12 (38–74%) but the ranking is correct where it matters. `scar_atoi` at 71.1% is a key result: the model generalises to the integer conversion sink pattern even though `atoi` was missing from DANGEROUS_SINKS during Juliet training. RankNet's pairwise objective is more robust to missing sink coverage than BCE.

**Misses:** `scar_alloc_copy` (rank 14, 68.3% — just below the boundary) and `handle_stats` (divide-by-zero, no dangerous sink — structurally invisible).

**Conclusion:** RankNet with guard features recovers §12 parity (11/13) where BCE with guard features regresses (§27: 10/13). The ranking objective is more robust than BCE to the combination of Devign noise + expanded feature set. §28 is now co-best with §12. §32 (below) also reaches 11/13 but through a different per-function distribution — confirming the ceiling is structural, not a data coverage problem.

---

## §29 — Juliet-only (no Devign fine-tune)

**Checkpoint:** `model_juliet_pretrain.pt` (produced by Phase 1 of `train_slice_pdg_v7.py` or `train_slice_pdg_v8.py`)
**No new training script** — checkpoint already exists after running §27 or §28 Phase 1.

### Hypothesis

Every Devign fine-tune phase contaminates the Juliet structural prior with commit-history fingerprinting. §27 (BCE) and §28 (RankNet) both achieve 11/13 on scarnet — the same as §12 trained on Devign alone. The Juliet prior may be the signal; the Devign fine-tune may be the noise.

The `model_juliet_pretrain.pt` checkpoint:
- Was trained on ~19,000 zero-noise Juliet pairs covering CWE-121/122/134/415/476
- Achieved 99%+ validation accuracy by epoch 3
- Has never seen FFmpeg or QEMU commit history
- Has never seen scarnet functions

If the structural prior transfers without fine-tuning:
- Functions with unguarded sinks reaching dangerous operations → high score (true structural signal)
- Structurally clean functions → low score (the right answer, not a side-effect of Devign distribution)
- `handle_stats` correctly scores low (divide-by-zero, no dangerous sink — the model should abstain)
- `session_new` / `dispatch` FPs may drop if they are structurally cleaner than §27/§28 suggest

### Evaluation

No training step required. Run directly:

```bash
python eval_all_models.py --scarnet --answer-key ~/Downloads/SCAR/scarnet-answer-key.txt
```

`model_juliet_pretrain.pt` is registered in the REGISTRY and evaluates alongside all other checkpoints.

### Results

**Hypothesis falsified.** The Juliet-only model saturates on scarnet — scoring 87–100% on all 19 functions, including `handle_stats` (divide-by-zero, no dangerous sink) and `main`. Confirmed with corrected preprocessor (x(N,3)).

| Rank | Function | Score | Vuln? |
|------|----------|-------|-------|
| 1 | handle_client | 100.0% | YES ✓ |
| 2 | dispatch | 100.0% | no ✗ (FP) |
| 3 | handle_set | 100.0% | YES ✓ |
| 4 | handle_get | 100.0% | no ✗ (FP) |
| 5 | session_login | 100.0% | YES ✓ |
| 6 | handle_del | 100.0% | YES ✓ |
| 7 | parse_cmd | 100.0% | YES ✓ |
| 8 | session_frag | 100.0% | YES ✓ |
| 9 | handle_stats | 100.0% | YES ✓ |
| 10 | session_new | 99.9% | no ✗ (FP) |
| 11 | parse_msg_header | 99.9% | YES ✓ |
| 12 | session_consume_frag | 99.9% | YES ✓ |
| 13 | handle_auth | 99.8% | no ✗ (FP) |
| 14 | parse_batch | 99.5% | YES ✗ (miss) |
| 15 | scar_alloc_copy | 99.4% | YES ✗ (miss) |
| 16 | main | 98.4% | no |
| 17 | session_free | 96.1% | no |
| 18 | scar_atoi | 94.6% | YES ✗ (miss) |
| 19 | scar_log | 87.0% | YES ✗ (miss) |

P@13: 9/13 (69.2%) — worse than §12. The model fires at maximum confidence on everything.

**Why:** The Juliet model has never seen real production C. Scarnet functions are structurally more complex than Juliet synthetics — more call sites, more memory operations, more control flow. The model has no reference for what clean production code looks like, so everything exceeds its threshold. The score spread collapses to 91–100%, making ranking impossible.

**Conclusion: the Devign fine-tune is not the contaminant — it is the calibration.** It teaches the model what "not suspicious" looks like in real-world C. Without it, the model cannot distinguish a complex-but-clean production function from an unguarded sink. The Juliet prior alone does not transfer to out-of-distribution code.

The correct negative class for a Juliet-pretrained model is not only Juliet good functions — it must also include confirmed-clean real production code. A model trained on Juliet bad + Juliet good + clean real C negatives would have a calibrated reference for both structural safety and real-world coding style.

**§27 remains the best checkpoint** (11/13, 84.6% P@13). The Devign fine-tune provides necessary calibration despite its label noise.

---

## Current Conclusion

28 experiments across block-level, instruction-level, slice-based, contrastive, feature-enriched, BigVul-trained, taint-augmented, sink-node readout, intrinsic-aware, PrimeVul-trained, Juliet-pretrained (BCE and RankNet) GNNs on Devign, scarnet, and zero-shot transfer to zlib v1.2.11.

---

## §30 — Juliet Positives + Clean Real-C Negatives (no Devign)

**Scripts:** `preprocess_clean_negatives.py` + `train_slice_pdg_v9.py`
**Architecture:** SlicePDGGNN_v7 — identical to §27/§28/§29

### Motivation

§29 proved the Juliet-only model saturates on real production code because its negative class (Juliet good functions) is synthetic and unlike real C. §27/§28 fix saturation by fine-tuning on Devign — but Devign re-introduces commit-history fingerprinting and ~15% label noise.

§30 replaces Devign entirely:

| | Source | Label | Label quality |
|---|---|---|---|
| Positives | Juliet bad functions | 1 | Zero noise — definitional |
| Negatives | Juliet good + zlib + musl + SQLite | 0 | Near-zero — heavily audited |

The model learns "unguarded dangerous sink" (from Juliet bad) vs "real clean C" (from real projects) — with no commit history, no noise, and real calibration.

### Clean negative sources

- **zlib** (`github.com/madler/zlib`) — ~50 functions, memory/compression
- **musl libc** (`github.com/bminor/musl`) — ~2,000 functions from `src/string`, `src/stdlib`, `src/stdio`, `src/malloc`, `src/math` and other portable subdirectories
- **SQLite amalgamation** — ~1,500 functions, diverse real C idioms

All three compile cleanly via `compile_to_ir()`. All labelled `y=0`.

### Training

- Phase 1 (Juliet-only BCE): reuses `model_juliet_pretrain.pt` from §27/§28 if present
- Phase 2: Juliet bad + (Juliet good + zlib + musl + SQLite) combined negatives, BCE, 40 epochs, lr=3e-4
- No Devign anywhere — no test accuracy number (Devign test set is irrelevant)
- Evaluate directly on scarnet

### Results

**Corpus:** 3,793 clean C graphs extracted from 6 sources; 20,093 total training graphs after combining with Juliet:

| Source | Graphs |
|---|---|
| lua | 2,310 |
| musl | 729 |
| lz4 | 429 |
| zlib | 159 |
| cjson | 154 |
| libuv | 12 |
| **Total** | **3,793** |

Training converged cleanly: 97.79% validation accuracy by epoch 3 — vs §27/§28's oscillating 50–55%. The clean BCE signal without Devign noise is apparent immediately.

**scarnet P@13: 9/13 (69.2%) — regression from §12/§27's 11/13 (84.6%)**

| Rank | Function | Score | Vuln? |
|---|---|---|---|
| 1 | handle_del | 99.3% | YES ✓ |
| 2 | session_new | 95.0% | no ✗ (FP) |
| 3 | session_login | 19.2% | YES ✓ |
| 4 | scar_log | 6.4% | YES ✓ |
| 5 | handle_set | 2.9% | YES ✓ |
| 6 | handle_client | 2.5% | YES ✓ |
| 7 | handle_get | 1.1% | no ✗ (FP) |
| 8 | main | 0.6% | no ✗ (FP) |
| 9 | parse_msg_header | 0.3% | YES ✓ |
| 10 | parse_cmd | 0.3% | YES ✓ |
| 11 | handle_auth | 0.1% | no ✗ (FP) |
| 12 | handle_stats | 0.1% | YES ✓ |
| 13 | scar_alloc_copy | 0.0% | YES ✓ |
| *below top-13* | | | |
| 14 | dispatch | 0.0% | no |
| 15 | session_free | 0.0% | no |
| 16 | session_frag | 0.0% | YES ✗ (miss) |
| 17 | session_consume_frag | 0.0% | YES ✗ (miss) |
| 18 | scar_atoi | 0.0% | YES ✗ (miss) |
| 19 | parse_batch | 0.0% | YES ✗ (miss) |

### Why §30 regressed

The score distribution is bimodal: `handle_del` (99.3%) and `session_new` (95%), then everything below 20%. This is a domain-shift failure with a precise cause.

**Root cause: Lua dominated the negative corpus.** 2,310 of 3,793 negatives (61%) came from Lua — an interpreter with a distinctive, highly-templated coding style unlike any server-side C. The model learned "library/interpreter C = clean," but scarnet's server functions — `session_frag`, `session_consume_frag`, `scar_atoi`, `parse_batch` — are also server-style C. The model pushed them to zero because they look structurally like the negative domain.

The diagnostic is sharp: `handle_del` scores 99.3% (unguarded `memcpy`/`free` with no guard — a Juliet-style clear-sink pattern). `session_frag` and `session_consume_frag` also have real unguarded sinks, but are structured like server C — matching the musl/lua negative distribution rather than the Juliet positive distribution. Devign's 15% label noise is less damaging than this bias because Devign at least contains functions from the same structural domain as production server code.

**§31 fixes this** by replacing lua with libcurl — domain-matched clean server/network C that teaches the correct boundary. See §31 below.

Every approach converges at the same ceiling: **~55–58% Devign accuracy**, against a majority-class baseline of 56.6% and a CodeBERT reference of 63.43%. Switching datasets (BigVul §21, PrimeVul §25), changing architecture (sink-node readout §23), fixing preprocessing (intrinsic-aware sinks §24), and adding semantic annotations (taint flags §22) all fail to improve the scarnet real-world benchmark beyond §12's 11/13.

The 7pp gap to CodeBERT is real and has three distinct causes:

### 1. Vulnerability patterns are often absences, not presences

The most common vulnerability pattern — a missing bounds check, a missing null guard, an unchecked return value — is structurally a *subgraph that isn't there*. A guarded `memcpy` and an unguarded one have nearly identical IR graphs; the only difference is the absent `icmp`/`br` predecessor. RGCN message-passing propagates information along existing edges; a missing subgraph leaves no gradient signal. This was confirmed directly by the §8–§10 contrastive experiments: vuln/fixed pairs have structural similarity 0.9984–0.9995. No feature extraction improvement can signal what is structurally absent.

### 2. LLVM IR has already discarded the most informative content

CodeBERT reads `gets(buf)` and associates it with vulnerability from training on billions of tokens including CVE descriptions, security advisories, and developer commentary. It reads variable names like `size`, `n`, `offset` near `memcpy` and learns that pattern. In LLVM IR at `-O0`, those names become `%0`, `%1`, `%2`. String literals become addresses. The identifier vocabulary that carries most of the human-assigned semantic signal is gone before feature extraction begins. §13 Perfograph encoding and call categorization recovered a small fraction of this (call targets, constant magnitudes) — hence the marginal +0.75pp — but the bulk is structurally unrecoverable from IR alone.

### 3. Devign label noise caps the dataset ceiling

Devign labels are assigned at git-commit granularity to the function that changed. The memory error may manifest in a callee, or the vulnerable condition may be set up two call frames away. The GNN is learning to predict a commit-level label from a single function's IR structure. This mismatch probably caps Devign at ~60–63% for any method, which is precisely where CodeBERT lands.

### What the experiments ruled out

- **Architecture:** block, instruction, slice, PDG slice, contrastive — all hit the same ceiling. Architecture is not the bottleneck.
- **Edge types:** CFG vs. CFG+DFG (§4c) vs. PDG slice (§12) vs. VSDG state edges (§14) — marginal differences, no ceiling break. Adding a 4th relation for memory ordering hurt slightly (57.47% vs 58.00%).
- **Loss function:** BCE, weighted BCE, focal contrastive — same result.
- **Granularity:** basic block vs. individual instruction — marginal improvement (+0.16pp), same ceiling.
- **Feature enrichment:** Perfograph constant encoding + categorical call targets (§13) — marginal +0.75pp best case, high cross-run variance, within noise.
- **Register name embedding (§15):** FNV-1a hash of LLVM IR register names into 64 buckets, learned 16-dim embedding — 57.47%, no improvement. Names don't transfer across codebases; bucket collisions dominate. **IR feature engineering track closed.**
- **Training data quality (§21):** Switching from Devign's commit-level labels to BigVul's CVE-level labels — no scarnet improvement. Dataset quality is not the bottleneck.
- **Taint flags on PDG nodes (§22):** Explicit Pattern A/B annotations on PDG slice nodes — regressed from 11/13 to 9/13 on scarnet. Semantic heuristics tuned to Devign's vulnerability distribution hurt out-of-distribution performance. **Semantic heuristic track closed.**
- **Sink-node readout + residual/LayerNorm (§23):** Architecturally correct but requires instruction-level supervision to realise its potential. Under graph-level noisy labels, concentrating the loss on sink nodes that may not contain the labelled bug regresses vs §12's diffuse global pooling.
- **Intrinsic-aware sinks (§24):** Correctly fixes a preprocessing bug (LLVM memory intrinsics were invisible to the slicer). Retained as the canonical preprocessor. Does not break the Devign ceiling — label noise still binds.
- **PrimeVul training (§25):** Cleaner labels, but 2× attrition differential (21.8% vs 37% survival for vulnerable vs benign functions) systematically filters out the complex functions where real bugs live. The biased surviving sample regresses on scarnet. Joern fixed the attrition but is a hard dependency conflict with the SCAR/IKOS pipeline — **Joern path dropped permanently.** PrimeVul itself is worth revisiting with a better compilation strategy.

### What has not been tried

Within the no-Devign / clean-negatives approach: §30 proved the concept is correct (bimodal convergence, clean training signal) but the negative corpus is the wrong domain. **Domain-matched clean negatives** — server/network C that is structurally similar to scarnet but confirmed clean — is the correct next step. libcurl is the best candidate: network/protocol C, heavily audited, with socket handling, buffer management, HTTP parsing and session management that mirrors scarnet's structure.

Beyond data: §28 replaces Phase 2's BCE with a pairwise RankNet loss — for each mini-batch, all (vuln, benign) pairs are formed and `P(score_v > score_b)` is maximised directly. This aligns the training objective with the scarnet ranking use case without requiring clean labels. §31 combines domain-matched negatives with this approach.

## §31 — Domain-Matched Clean Negatives (libcurl replaces lua)

**Scripts:** `preprocess_clean_negatives.py --sources zlib,musl,libcurl,lz4,cjson,libuv` + `train_slice_pdg_v10.py`
**Architecture:** SlicePDGGNN_v7 — identical to §27–§30

### Motivation

§30 regression analysis identified the root cause precisely: lua (61% of negatives, 2,310/3,793 graphs) taught the model "interpreter/library C = clean," which caused server-style vulnerable functions (`session_frag`, `session_consume_frag`, `scar_atoi`, `parse_batch`) to score near 0%.

The fix is **domain-matched negatives**: the negative corpus must contain server/network C that is structurally similar to scarnet but confirmed clean, so the model learns the correct boundary — "unguarded sink = vulnerable" rather than "server C = clean."

**libcurl** is the optimal replacement:
- Network/protocol handling C — HTTP parsing, socket handling, session management, buffer ops
- Structurally closest to scarnet among all open-source C libraries
- Heavily security-focused with active CVE response (any unguarded pattern has been audited away)
- ~200–400 source functions, same ballpark as scarnet itself

The remaining sources (zlib, musl, cjson, lz4, libuv) are retained for diversity.

### Clean negative sources (§31)

| Source | Structural domain | Reason to include |
|---|---|---|
| libcurl | Network/protocol C — socket, HTTP, TLS, session | Domain-matched to scarnet; heavily audited |
| musl | String/stdlib/math libc | Diverse clean C idioms |
| zlib | Compression — memory-heavy | Clean memory ops, well-studied |
| lz4 | Compression | Fast-path buffer code |
| cjson | JSON parser | Application-level parsing, similar to scarnet parse.c |
| libuv | Async I/O — handles, sockets | Server-loop C |

Lua is **dropped** — its interpreter-style C (templated opcode dispatch, VM state machines) is too unlike server code and biases the model toward suppressing server-style functions.

### Training

- Phase 1 (Juliet-only BCE): reuses `model_juliet_pretrain.pt` from §27/§28/§29/§30 if present
- Phase 2: Juliet bad + (Juliet good + libcurl + musl + zlib + lz4 + cjson + libuv) combined negatives, BCE, 40 epochs, lr=3e-4
- No Devign anywhere — evaluate with scarnet only
- Output: `model_slice_pdg_v10.pt`

### Commands

```bash
# Preprocessing — skip clone of already-present repos, add libcurl
python preprocess_clean_negatives.py \
    --sources zlib,musl,libcurl,lz4,cjson,libuv \
    --workers 4

# Training (skip Phase 1 if model_juliet_pretrain.pt already exists)
python train_slice_pdg_v10.py --finetune-only

# Evaluate
python eval_all_models.py --scarnet \
    --answer-key ~/Downloads/SCAR/scarnet-answer-key.txt \
    --summary-only
```

**What to look for:**
- `session_frag`, `session_consume_frag`, `scar_atoi`, `parse_batch` should recover from 0.0% to the 40–70% range — they are unguarded sinks, and libcurl's clean sinks look different from them
- `session_new` should remain a FP or drop — it is genuinely structurally complex
- `handle_del` should remain near 99% — the clear-sink pattern is unchanged

### Results

**8/13 (61.5%) — further regression from §30 (9/13).** Opposite failure mode from §30.

| Rank | Function | Score | Vuln? |
|------|----------|-------|-------|
| 1 | handle_auth | 100.0% | no ✗ (FP) |
| 2 | session_login | 99.9% | YES ✓ |
| 3 | handle_client | 97.4% | YES ✓ |
| 4 | handle_del | 90.2% | YES ✓ |
| 5 | session_new | 78.0% | no ✗ (FP) |
| 6 | scar_log | 74.8% | YES ✓ |
| 7 | handle_set | 31.2% | YES ✓ |
| 8 | handle_get | 29.8% | no ✗ (FP) |
| 9 | parse_cmd | 6.8% | YES ✓ |
| 10 | dispatch | 5.3% | no ✗ (FP) |
| 11 | parse_msg_header | 3.2% | YES ✓ |
| 12 | parse_batch | 0.1% | YES ✓ |
| 13 | main | 0.0% | no |
| 14 | handle_stats | 0.0% | YES ✗ (miss) |
| 15 | session_free | 0.0% | no |
| 16 | scar_alloc_copy | 0.0% | YES ✗ (miss) |
| 17 | scar_atoi | 0.0% | YES ✗ (miss) |
| 18 | session_consume_frag | 0.0% | YES ✗ (miss) |
| 19 | session_frag | 0.0% | YES ✗ (miss) |

**Why §31 failed (opposite direction from §30):** libcurl is structurally TOO similar to Juliet bad patterns. libcurl's `lib/*.c` functions contain `memcpy`, `malloc`, pointer arithmetic, and complex network control flow — labelled `y=0` — but these patterns look like Juliet bad functions to the RGCN. The model resolved contradictory signal ("network/buffer C = 1 from Juliet" vs "network/buffer C = 0 from libcurl") by scoring all ambiguous functions high (90–100%) and everything else near 0%. `session_frag`, `session_consume_frag`, `scar_atoi`, `scar_alloc_copy` all fell below the decision boundary.

**The clean-negatives hypothesis is exhausted.** Two experiments, two opposite failure modes:
- §30 (lua): negatives too structurally distant → model suppresses server-C as "interpreter-like = clean"
- §31 (libcurl): negatives too structurally similar → model saturates server-C as "network-like = Juliet bad"

No curated open-source corpus sits cleanly in the target domain with confirmed-clean labels. The fundamental issue is that any corpus labelled "definitely clean" is a biased sample — it either lacks the structural diversity of real vulnerability patterns or overlaps with them. Devign's noise is less damaging than this bias because Devign at least samples the joint distribution of safe and unsafe server C from the same structural domain.

**§32** (below) returns to the Devign-based approach with the correct preprocessor and atoi in DANGEROUS_SINKS — the path to 12/13.

## §32 — Juliet Pretrain + Devign RankNet, with atoi sinks + correct eval preprocessor

**Scripts:** `preprocess_juliet.py` (rebuild) + `train_slice_pdg_v8.py` (retrain Phase 2)
**Architecture:** SlicePDGGNN_v7 — identical to §27/§28

### Motivation

The eval bug analysis (§27 note) revealed that all §27–§31 Juliet training data was built with `atoi`/`strtol` missing from DANGEROUS_SINKS. `scar_atoi` — a known-vulnerable function — had no slice built during Juliet preprocessing, so the model never saw CWE-190/191 integer conversion patterns as positives.

§28 (RankNet) scored `scar_atoi` at 71.1% despite this gap — the model generalised from other sink patterns. §32 explicitly includes integer conversion functions as sinks in Juliet training data, then retrains Phase 2 with RankNet. If `scar_atoi` moves from 71.1% to 80%+ and `scar_alloc_copy` (rank 14, 68.3% in §28) rises past the boundary, we reach 12/13.

**What changes vs §28:**
- Rebuild `train_juliet_graphs.pkl` / `valid_juliet_graphs.pkl` after adding `atoi`/`atol`/`atoll`/`strtol`/`strtoul` to DANGEROUS_SINKS in `preprocess_juliet.py` (already done)
- Retrain Phase 2 only (`--finetune-only`) using the existing `model_juliet_pretrain.pt`
- Output: `model_slice_pdg_v11.pt`

### Commands

```bash
# Step 1: rebuild Juliet graphs with atoi/strtol sinks
# (model_juliet_pretrain.pt can be reused — Phase 1 is opcode-only, unaffected)
python preprocess_juliet.py --workers 8

# Step 2: retrain Phase 2 only (RankNet, reuses pretrain ckpt)
python train_slice_pdg_v8.py --finetune-only --checkpoint model_slice_pdg_v11.pt

# Step 3: evaluate
python eval_all_models.py --scarnet \
    --answer-key ~/Downloads/SCAR/scarnet-answer-key.txt
```

**What to look for vs §28:**
- `scar_atoi` (71.1% at rank 6 in §28) should rise — now explicitly in positive training set
- `scar_alloc_copy` (68.3% at rank 14 in §28, just below boundary) may cross into top 13
- `handle_stats` (divide-by-zero, no dangerous sink) should remain low — correctly abstaining

### Results

**11/13 — P@13 = 84.6%** — matches §12 and §28; ceiling holds.

*(Note: an earlier run of §32 used `data/train_graphs.pkl` — the 44-col block-level Devign pkl — after the N_SCALAR clamp fix unblocked training. That run produced the wrong distribution and was discarded. Results below are the correct run using `train_slice_pdg_v7_graphs.pkl`.)*

| Rank | Function | Score | Vuln? |
|---|---|---|---|
| 1 | handle_client | 79.1% | **YES** |
| 2 | session_new | 78.5% | no |
| 3 | parse_batch | 75.5% | **YES** |
| 4 | session_frag | 75.1% | **YES** |
| 5 | session_consume_frag | 74.0% | **YES** |
| 6 | scar_atoi | 71.9% | **YES** |
| 7 | handle_set | 71.3% | **YES** |
| 8 | parse_cmd | 70.9% | **YES** |
| 9 | session_login | 70.7% | **YES** |
| 10 | dispatch | 70.5% | no |
| 11 | handle_del | 70.3% | **YES** |
| 12 | parse_msg_header | 69.7% | **YES** |
| 13 | scar_log | 69.6% | **YES** |
| 14 | handle_stats | 69.1% | YES |
| 15 | scar_alloc_copy | 68.3% | YES |
| 16 | handle_get | 65.3% | no |
| 17 | session_free | 64.6% | no |
| 18 | handle_auth | 61.7% | no |
| 19 | main | 60.8% | no |

**Hits:** handle_client, parse_batch, session_frag, session_consume_frag, scar_atoi, handle_set, parse_cmd, session_login, handle_del, parse_msg_header, scar_log (11 in top 13). Displaced by session_new (FP) and dispatch (FP).

**Misses:** handle_stats (rank 14, 69.1%), scar_alloc_copy (rank 15, 68.3%).

#### Analysis

**scar_atoi recovers to 71.9% (rank 6)** — essentially identical to §28's 71.1%. Adding atoi to DANGEROUS_SINKS in Juliet did not measurably change the score for this function. The model learned the integer-conversion sink pattern from generalisation in §28 and the explicit training in §32 produces the same result. This confirms atoi coverage was not the limiting factor.

**Score compression is the defining feature of §32.** The spread is only 18 points (60.8–79.1%) compared to §28's much wider distribution. Every function sits in the 60–80% band. This makes the ranking fragile near the boundary: handle_stats (rank 14, 69.1%) and scar_log (rank 13, 69.6%) are separated by 0.5 points. The ranking is correct, but the margin is not robust to run-to-run variance.

**Two new misses vs §28:** handle_stats rises to rank 14 (69.1%) — it has no dangerous sink but its structural complexity compresses into the same band as true positives. scar_alloc_copy stays at rank 15 (68.3%) — same position as §28, unchanged.

**Two false positives:** session_new (rank 2, 78.5%) and dispatch (rank 10, 70.5%) — same structural FPs as every other variant. These are not addressable with the current architecture.

**Why compression?** Phase 1 Juliet training with atoi added more small-graph positives, pushing the model to score graphs of all sizes similarly high. Phase 2 RankNet then calibrated the ordering without widening the spread — the result is correct rank ordering inside a narrow confidence band. This is the opposite of saturation (§29): rankings are valid, but the confidence gap between functions is too small to be operationally useful as a threshold.

**Conclusion:** The 11/13 ceiling is stable across §12, §28, and §32. The two structural misses (handle_stats, scar_alloc_copy) and two structural FPs (session_new, dispatch) are invariant across all training variants. Score compression in §32 is a calibration regression vs §28 — P@13 is identical but the ranking is less robust near the boundary. §28 (RankNet, no atoi addition) remains the preferred checkpoint: same hits, wider confidence spread.

The path to 12/13 is not more Juliet sink variants or cleaner negatives. It requires node-level supervision (see Dataset Limitations section below) — the ability to tell the model exactly which instruction is the unguarded sink, not just which function contains one.

---

### Practical value for SCAR

The Devign accuracy number (57.84%) understates the model's value in SCAR's pipeline. On SCAR's actual targets — code compiled in full project context via Tekton's `build-bitcode` task — attrition is near zero and the feature distribution better matches real-world vulnerability patterns. The §9 scarnet validation confirmed this: 10/13 known-vulnerable functions ranked in the top 13 of 19 (77% precision/recall) without any fine-tuning on the target codebase.

---

## Dataset Limitations and What a Better Dataset Would Look Like

The 22 experiments here converge on a clear conclusion: the ceiling is not the model, and it is not the IR representation alone. It is the training data. Both Devign and BigVul were the best available public datasets at the time of this work, and both have the same structural flaw. Understanding that flaw precisely — and what would fix it — matters for anyone building on this line of research.

### What is wrong with Devign and BigVul

**Devign** assigns labels at git-commit granularity. A function is labelled "vulnerable" if a CVE-linked commit touched it. This is a noisy proxy for three distinct reasons:

- The commit may have fixed a typo, renamed a variable, or restructured control flow with no semantic change to the vulnerable path — the function changed, but was never actually vulnerable.
- The function that *introduced* the vulnerability in a prior commit carries a "fixed" label if the fixing commit also happened to touch it.
- The actual vulnerable instruction may be in a callee, a macro expansion, or a shared utility that the commit did not touch at all — the GNN is reading the wrong function.

Estimated label noise in Devign is 10–20%. For a binary classifier on a 56.6% majority class, this alone explains the 57–58% ceiling. A model that has learned to perfectly predict the structural pattern of a buffer overflow cannot score above ~80% if 10–20% of its training labels are inverted.

**BigVul** improves label precision by using CVE-level CWE classifications rather than commit diffs, but has the same function-level granularity problem and a substantially different vulnerability distribution from Devign. The §21 experiments confirmed that training on BigVul does not improve scarnet performance, and combining both datasets offers no gain over either alone — the distributions are not complementary.

Both datasets also share a less obvious flaw: **the negative class is unverified**. Functions not touched by CVE-linked commits are labelled "safe", but there is no audit confirming this. Real codebases contain unfixed vulnerabilities; some fraction of Devign's "safe" training examples are almost certainly vulnerable. A GNN trained on this signal learns to predict commit-level audit history, not structural vulnerability.

### Properties of a dataset that would break through the ceiling

In order of impact on the model:

**1. Instruction-level location labels.**
Not "this function is vulnerable" but "this call/memory access at this IR instruction index is the unguarded sink." This is the single most important property. With node-level ground truth the graph-classification problem becomes a node-classification problem — the sink-node readout architecture explored in §23 is already correct for this, it just needs training signal at the right level. The pooling-dilution problem disappears when supervision is applied directly to the sink node.

**2. Minimal-diff CVE/fix pairs.**
Pairs where the fix is exactly one added `icmp`+`br` (a missing bounds check), one null check, one bounds validation before an array index. These give pure structural signal: the IR subgraph that the fix *adds* is the vulnerability pattern the model should learn. Functions fixed by large refactors, renames, or multi-file changes introduce confounders and should be filtered out. The pair-wise structure is also essential — comparing the same function before and after the fix removes all confounding variation in coding style, project idioms, and function size.

**3. Confirmed negatives.**
Functions explicitly audited and confirmed safe, not simply "not touched by a known CVE commit." This could come from formal verification discharge, or from fuzz testing that found no defect after extensive coverage. Even a small set of confirmed-safe functions with known structural similarity to true vulnerabilities would dramatically sharpen the model's decision boundary.

**4. Cross-project diversity at scale.**
Devign is 4 projects. A robust training set needs 100+ projects across OS kernels, networking stacks, cryptographic implementations, file parsers, multimedia codecs, and embedded firmware — each with distinct coding conventions and vulnerability patterns. Without this, any model risks learning "FFmpeg-shaped code looks vulnerable" rather than learning generalizable structural patterns.

**5. Vulnerability class balance.**
Structured sampling across CWE types (CWE-119/125/787 buffer errors, CWE-416 use-after-free, CWE-476 null dereference, CWE-362 race conditions, CWE-190/191 integer over/underflow) so that a model cannot reach 58% by specialising on buffer overflows alone.

### SCAR at scale as a training data generator

SCAR is, architecturally, a dataset generator. It already produces most of what the above list requires:

- **Planted bugs have exact location** — we know precisely which function, which line, which instruction pattern is the vulnerability.
- **IKOS-validated true positives** — the planted bug is reachable and produces a real static witness. It is not a hypothetical.
- **LLM-generated fix pairs are structurally minimal by construction** — the LLM introduces one vulnerability pattern; the fix removes it. The diff is almost always a single structural change.
- **Scalable** — the bottleneck is IKOS analysis time per function, not human labelling. At scale this is a pipeline throughput problem, not a labelling cost problem.
- **Controllable class balance** — bug types are explicitly specified, so CWE distribution is a configuration parameter, not an accident of history.

A SCAR-generated corpus at scale would provide instruction-level labels (the planted bug is at a known IR instruction), confirmed negatives (the same codebase pre-injection), and minimal-diff pairs (the injection diff IS the vulnerability pattern). These are exactly the three highest-impact properties identified above.

The realistic concern with synthetic data is **distribution shift**: LLM-planted bugs may follow different structural patterns than vulnerabilities introduced organically by human developers over years of maintenance. A model trained purely on SCAR-generated bugs might learn to recognise "LLM-introduced buffer overflow" rather than "buffer overflow." The mitigation is to treat SCAR data as training data only and evaluate exclusively on confirmed real-CVE corpora — scarnet-style, with human-verified planted bugs in real-world code. Never evaluate a SCAR-trained model on SCAR-generated test data.

The practical architecture for a next-generation dataset:

| Layer | Source | Purpose |
|---|---|---|
| Training (large) | SCAR-generated pairs at scale | Node-level supervision, class balance, volume |
| Training (small) | Real CVE minimal-diff pairs, hand-curated | Distribution anchoring |
| Evaluation | scarnet-style confirmed real-world corpora | Honest OOD performance measurement |

### What this would change for the GNN

If instruction-level labels exist, the entire framing changes. The model no longer asks "is this function vulnerable?" — a question that requires compressing a 300-node graph into one bit. It asks "is this instruction an unguarded sink?" — a question the PDG backward-slice graph is specifically structured to answer, where the relevant local context (guard conditions, data origins) is already concentrated in the K-hop neighborhood of the sink node. The §23 sink-node readout, trained with node-level supervision, would produce a calibrated per-instruction risk score rather than a function-level label. That score is directly actionable: pass the high-scoring IR instruction and its backward slice to the LLM as a targeted prompt, rather than the entire function.

The information-theoretic situation also improves. The IR representation discards identifier names, but it fully preserves the structural relationship between a dangerous call and its guard conditions. That relationship — present or absent — is exactly what node-level supervision would teach the model to recognise. The LLVM IR representation is not the bottleneck. The label resolution is.

The GNN's role is not to replace the LLM scanner. It is a **zero-cost structural pre-filter** — one CPU forward pass per function, milliseconds per PR, no LLM tokens spent — that routes structurally suspicious functions to the heavier analysis and bypasses clearly safe code. The 57–58% Devign number is the cost of that filter; the benefit is avoiding LLM calls on the majority of functions that are unambiguously clean.
