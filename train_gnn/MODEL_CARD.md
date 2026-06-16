---
language: c
license: mit
tags:
  - vulnerability-detection
  - graph-neural-network
  - llvm-ir
  - static-analysis
  - security
datasets:
  - devign
metrics:
  - accuracy
---

# scar-gnn-defect-detector

A collection of lightweight graph neural networks that classify C functions as vulnerable
or safe using their LLVM IR graphs. Designed as a **zero-cost pre-filter** for LLM-based
vulnerability triage pipelines.

## Models

| File | Architecture | Devign test accuracy |
|---|---|---|
| `model.pt` | DefectGNN block-level (§4d, 60ep h=128, 2 relations) | **55.52%** |
| `model_instr.pt` | InstrGNN instruction-level (§7, 30ep h=64, 3 relations) | **56.53%** |
| `model_slice.pt` | SliceGNN DFG backward slice (§11, 30ep h=64, 3 relations) | **56.64%** |
| `model_slice_pdg.pt` | SlicePDGGNN PDG slice (§12, 30ep h=64, 3 relations) | **56.48%** |

`model.pt` is the recommended checkpoint for the ranker pipeline — it uses the 45-feature
block-level representation and runs fastest at inference. `model_instr.pt` achieves the
highest Devign accuracy.

### Latest experiments (§13–§15, not yet uploaded)

§13 added Perfograph constant encoding (`sign(C)*log2(|C|+1)`) and categorical call targets
(5 danger categories: Alloc/Copy/String/FileIO/Network) to the instruction-level model.
Best single-run result: **58.75%** (`train_instr_v2.py`). High cross-run variance (~54–59%)
at the ~1,250-sample split scale makes this a noisy improvement over §7's 56.53%.

§14 adds VSDG memory ordering edges (load/store pairs on the same pointer → directed state
edge, type=3, `num_relations` 3→4). Result: **57.47%** — below §7 baseline. State edges
add density without benefit at Devign scale; val accuracy peaked lower (56.97% vs 59.44%
in §13), confirming no improvement. Scripts: `preprocess_instr_v3.py`, `train_instr_v3.py`.

§15 adds register name embedding: at `-O0`, clang preserves source variable names in IR
register names (`%buf.addr`, `%size`, `%cmp`). FNV-1a hashes them into 64 buckets; a
learned `nn.Embedding(65, 16)` gives a 16-dim name signal per node (conv1 input: 129→145).
Result: **57.47%** — identical to §14. Names do not generalize across Devign's four
codebases; bucket collisions dominate the signal. **IR feature engineering track closed.**
Scripts: `preprocess_instr_v4.py`, `train_instr_v4.py`.

## Model Description

### model.pt — DefectGNN (block-level)

- **Architecture:** two `RGCNConv` layers (2 relation types: CFG, DFG),
  AttentionalAggregation readout, two-layer MLP classifier
- **Input:** LLVM IR compiled with `clang -O0 -fno-inline -S -emit-llvm`; each basic
  block becomes a node with 45 semantic features (opcode distribution, branch density,
  memory op ratio, call density, phi count, block size)
- **Output:** Probability score ∈ [0, 1] that the function is vulnerable
- **Parameters:** ~293 KB (`hidden=128`)

### model_instr.pt / model_slice.pt / model_slice_pdg.pt — Instruction-level GNNs

- **Architecture:** Embedding lookup (vocab=110 opcodes) → two `RGCNConv` layers
  (3 relation types: CFG, DFG, call) → AttentionalAggregation → binary classifier
- **Input:** same IR compilation; each instruction becomes a node (opcode ID)
- **Parameters:** ~256 KB (`hidden=64`)

## Training

| Dataset | Split | Functions |
|---|---|---|
| Devign (FFmpeg, QEMU, Linux, LibreSSL) | train | ~10,100 |
| Devign | validation | ~1,252 |
| Devign | test | ~1,250 |

Training: Adam lr=1e-3, StepLR decay (γ=0.5, step=10).

## Performance

| Setting | Accuracy |
|---|---|
| Majority-class baseline | 56.60% |
| **model.pt — DefectGNN block-level** | **55.52%** |
| **model_instr.pt — InstrGNN instruction-level** | **56.53%** |
| **model_slice.pt — SliceGNN DFG slice** | **55.60%** |
| **model_slice_pdg.pt — SlicePDGGNN PDG slice** | **56.48%** |
| CodeBERT (source text) | 63.43% |

All results on Devign test set. The 1.24 pp gap over baseline is modest on Devign's
balanced test set. On real-world code the **ranking behaviour** matters more than binary
accuracy: the model assigns meaningfully higher scores to vulnerable functions, making it
useful as a ranker even when it does not clear a hard decision threshold.

### Real-world validation: scarnet

Applied to `johwes/scarnet` (a small intentionally-vulnerable C server, 19 functions
across 5 source files, 13 known-vulnerable):

| Metric | Value |
|---|---|
| Known-vulnerable functions in top-13 of 19 | **10 / 13 (77%)** |
| Precision at top-13 | **77%** |
| Recall at top-13 | **77%** |

Compilation flag used: `clang -O0 -fno-inline -S -emit-llvm` (required — `-O1` inlines
small functions into their callers, hiding them from per-function analysis).

**False negatives (4):** `scar_atoi`, `handle_set`, `session_consume_frag`, `handle_del`.
All are semantic bugs with no distinct structural IR signature — wrong argument, wrong
comparison operand, or off-by-one in a constant. These are LLM domain, not GNN domain.

**False positives (4):** `main`, `dispatch`, `handle_get`, `session_new` — structurally
complex functions that score high but are not vulnerable. All are immediately dismissible
in a one-sentence LLM triage step.

## Intended Use

This model is a **zero-cost ranker**, not a hard gate. Recommended pipeline:

```
clang -O0 -fno-inline -S -emit-llvm src/*.c -I include/ -o fn.ll
python scan_ir.py fn.ll --all-functions --threshold 0.5
# → ranked list of functions by vulnerability score
# → feed top-N to LLM for semantic triage
```

Use it to decide *which functions to show an LLM*, not to make final vulnerability
decisions. The LLM handles the semantic bugs the GNN misses.

## Limitations

- **Topology-only:** node features are opcode categories and structural metrics;
  identifier names, string literals, and type tokens are discarded. Semantic bugs
  (wrong comparison operator, wrong format string, off-by-one in a constant) produce
  identical IR topology to correct code and are undetectable.
- **Block-level granularity** (model.pt): each basic block is one node. Fine for
  function-level ranking; not suitable for pinpointing the exact buggy line.
- **GNN ceiling ~57–58%:** fifteen experiments across block-level, instruction-level,
  slice variants, Perfograph constant encoding, categorical call targets, VSDG memory
  edges, and register name embedding all converged in this range. The ceiling is
  representational — LLVM IR opcode graphs discard the identifier names and string
  literals that distinguish vulnerable from safe code. CodeBERT on source text reaches
  63.43%. IR feature engineering track is closed. See `docs/ir-embed.md §
  "Current Conclusion"` for the full analysis.
- **Devign distribution:** trained on C from large open-source projects; may not
  generalise well to embedded, kernel, or heavily macro-expanded code.

## Repository & Reproducibility

Source code, training scripts, and full experiment log (§1–§15):
**[johwes/llvm-ir-vuln-gnn](https://github.com/johwes/llvm-ir-vuln-gnn)**

Key files:
- `train_gnn/train.py` / `train_v2.py` — block-level (§4d / §13)
- `train_gnn/train_instr.py` / `train_instr_v2.py` / `train_instr_v3.py` / `train_instr_v4.py` — instruction-level (§7 / §13 / §14 / §15)
- `train_gnn/train_slice.py` / `train_slice_pdg.py` — slice variants (§11 / §12)
- `train_gnn/preprocess.py` / `preprocess_v2.py` — block-level graph extractors
- `train_gnn/preprocess_instr.py` / `preprocess_instr_v2.py` / `preprocess_instr_v3.py` / `preprocess_instr_v4.py` — instruction-level extractors
- `train_gnn/scan_ir.py` — inference CLI
- `docs/ir-embed.md` — full experiment log and current conclusion

## Citation

If you use these models, please cite the SCAR IR GNN repository:

```
@misc{scar-ir-gnn-2026,
  title  = {SCAR GNN Defect Detector},
  author = {johnnywesterlund},
  year   = {2026},
  url    = {https://huggingface.co/johnnywesterlund/scar-gnn-defect-detector}
  source = {https://github.com/johwes/llvm-ir-vuln-gnn}
}
```
