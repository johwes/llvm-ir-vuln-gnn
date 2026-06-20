---
language: code
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

## Recommended model

**`model_slice_pdg.pt`** (§12 PDG slice GNN) — best real-world recall across all 27
experiments. Ranks 11 of 13 known-vulnerable scarnet functions in the top 13 of 19
(84.6% recall@13). Devign test accuracy: 56.48%.

```bash
python scan_ir.py fn.ll --all-functions --threshold 0.5
python scan_ir.py fn.ll --context        # include PDG slice vulnerability context
```

## Models

| File | Section | Architecture | Devign |
|---|---|---|---|
| `model.pt` | §4d | DefectGNN block-level (60ep h=128, 2 rel) | 55.52% |
| `model_instr.pt` | §7 | InstrGNN instruction-level baseline (30ep h=64) | 56.53% |
| `model_instr_v2.pt` | §13 | Perfograph + call categories | 58.75% |
| `model_instr_v3.pt` | §14 | VSDG memory ordering edges | 57.47% |
| `model_instr_v4.pt` | §15 | Register name embedding | 57.47% |
| `model_instr_v5.pt` | §16 | Static analysis flags | 57.15% |
| `model_instr_v6.pt` | §17 | Taint propagation | 58.00% |
| `model_slice.pt` | §11 | DFG backward slice | 55.60% |
| **`model_slice_pdg.pt`** | **§12** | **PDG slice — recommended** | **56.48%** |
| `model_slice_pdg_v2.pt` | §22 | PDG + taint flags | — |
| `model_slice_pdg_v3.pt` | §23 | PDG sink-node readout + residual/LN | 55.40% |
| `model_slice_pdg_v4.pt` | §24 | PDG + intrinsic-aware sinks (retrain) | 55.00% |
| `model_slice_pdg_v5.pt` | §25 | PDG slice trained on PrimeVul | 55.56% |
| `model_slice_pdg_v6.pt` | §26 | PDG slice trained on Joern PrimeVul (VOCAB_SIZE=16) — dropped, see note | — |
| `model_slice_pdg_v7.pt` | §27 | PDG slice, Juliet pretrain + Devign BCE fine-tune; multi-feature x(N,3) | 56.12% |
| `model_slice_pdg_v8.pt` | §28 | PDG slice, Juliet pretrain + Devign RankNet fine-tune; pairwise ranking loss | TBD |
| `model_juliet_pretrain.pt` | §29 | Juliet-only — no Devign fine-tune; pure structural prior (no training on commit history) | — |
| `model_slice_pdg_v9.pt` | §30 | Juliet positives + clean real-C negatives (zlib/musl/SQLite); no Devign | — |

`model_bigvul_cls.pt` and `model_bigvul_combined.pt` (§21) are trained on BigVul only
and have no Devign score. See scarnet table below.

## Real-world validation: scarnet

Applied to `johwes/scarnet` (19 functions, 13 known-vulnerable). Evaluated all
checkpoints at top-13-of-19:

| Checkpoint | Section | Devign | Hits | P@13 | R@13 |
|---|---|---|---|---|---|
| model.pt | §4d block-level DefectGNN | 55.52% | 9/13 | 69.2% | 69.2% |
| model_instr.pt | §7 instr baseline | 56.53% | 9/13 | 69.2% | 69.2% |
| model_instr_v2.pt | §13 Perfograph + call categories | 58.75% | 10/13 | 76.9% | 76.9% |
| model_instr_v3.pt | §14 VSDG memory ordering edges | 57.47% | 10/13 | 76.9% | 76.9% |
| model_instr_v4.pt | §15 register name embedding | 57.47% | 8/13 | 61.5% | 61.5% |
| model_instr_v5.pt | §16 static analysis flags | 57.15% | 8/13 | 61.5% | 61.5% |
| model_instr_v6.pt | §17 taint propagation | 58.00% | 10/13 | 76.9% | 76.9% |
| model_slice.pt | §11 DFG slice | 55.60% | 10/13 | 76.9% | 76.9% |
| **model_slice_pdg.pt** | **§12 PDG slice** | **56.48%** | **11/13** | **84.6%** | **84.6%** |
| model_slice_pdg_v2.pt | §22 PDG + taint flags | — | 9/13 | 69.2% | 69.2% |
| model_slice_pdg_v3.pt | §23 PDG sink-node readout | 55.40% | 9/13 | 69.2% | 69.2% |
| model_slice_pdg_v4.pt | §24 PDG + intrinsic-aware sinks | 55.00% | 10/13 | 76.9% | 76.9% |
| model_slice_pdg_v5.pt | §25 PDG slice (PrimeVul training) | 55.56% | 9/13 | 69.2% | 69.2% |
| model_slice_pdg_v7.pt | §27 Juliet pretrain + Devign BCE FT | 56.12% | **11/13** | **84.6%** | **84.6%** |
| model_slice_pdg_v8.pt | §28 Juliet pretrain + Devign RankNet | 44.52% | 11/13 | 84.6% | 84.6% |
| model_juliet_pretrain.pt | §29 Juliet-only, no Devign FT | — | 9/13† | 69.2% | 69.2% |
| model_bigvul_cls.pt | §21 BigVul classifier | — | 9/13 | 69.2% | 69.2% |
| model_bigvul_combined.pt | §21 BigVul+Devign combined | — | 9/13 | 69.2% | 69.2% |
| ENSEMBLE (max) | all models | — | 9/13 | 69.2% | 69.2% |
| ENSEMBLE (mean) | all models | — | 9/13 | 69.2% | 69.2% |

†§29 saturates: scores 91.7–100% on all 19 functions including benign ones — no discrimination possible. Devign fine-tune provides necessary calibration; Juliet-only model has no reference for what real production C looks like.

**§12 is the uniquely best checkpoint.** Every other model and the full ensemble top out
at 10/13. The ensemble scoring 9/13 — worse than the best individual model — indicates
correlated errors: models trained on the same Devign distribution share the same failure
modes, so ensembling amplifies rather than cancels noise.

**Two irreducible misses (2/13):** no model catches all 13. The two remaining misses are
semantic bugs with no distinct structural IR signature — wrong comparison operand, or an
off-by-one in a constant. These are in the LLM triage domain, not the GNN domain.

**False positives (common):** `main`, `dispatch`, `handle_get`, `session_new` — structurally
complex functions that score high but are safe. Dismissible in a one-sentence LLM triage step.

## Model Description

### model.pt — DefectGNN (block-level)

- **Architecture:** two `RGCNConv` layers (2 relation types: CFG, DFG),
  AttentionalAggregation readout, two-layer MLP classifier
- **Input:** LLVM IR compiled with `clang -O0 -fno-inline -S -emit-llvm`; each basic
  block becomes a node with 45 semantic features (opcode distribution, branch density,
  memory op ratio, call density, phi count, block size)
- **Output:** probability ∈ [0, 1] that the function is vulnerable
- **Parameters:** ~293 KB (`hidden=128`)

### Instruction-level GNNs (§7–§17)

- **Architecture:** Embedding lookup (vocab=110 opcodes) → two `RGCNConv` layers
  (3 relation types: CFG, DFG, context) → AttentionalAggregation → binary classifier
- **Input:** same IR; each instruction becomes a node (opcode ID)
- **Parameters:** ~256 KB (`hidden=64`)

### PDG slice GNNs (§11–§23)

- **Architecture:** same embedding + RGCN backbone; input is a PDG backward slice
  from dangerous sinks (DFG + control dependence), not the full function graph
- **§12 (recommended):** AttentionalAggregation global readout
- **§23:** Sink-node readout (scatter-max over identified sink embeddings) + residual
  connections + LayerNorm — correct architecture for node-level supervision; 55.40%
  on Devign, 9/13 scarnet under graph-level noisy labels

## Training

| Dataset | Split | Functions |
|---|---|---|
| Devign (FFmpeg, QEMU, Linux, LibreSSL) | train | ~10,125 |
| Devign | validation | ~1,254 |
| Devign | test | ~1,251 |

Training: Adam lr=1e-3, StepLR decay (γ=0.5, step=10), 30 epochs, hidden=64.

## Experiment log summary (§1–§23)

| Section | Change | Devign | Scarnet |
|---|---|---|---|
| §4d | Block-level, 45 features, 2 relations | 55.52% | 9/13 |
| §7 | Instruction-level baseline (opcode only) | 56.53% | 9/13 |
| §11 | DFG backward slice from sinks | 55.60% | 10/13 |
| **§12** | **PDG slice (DFG + control dependence)** | **56.48%** | **11/13** |
| §13 | Perfograph encoding + call categories | 58.75%† | 10/13 |
| §14 | VSDG memory ordering edges | 57.47% | 10/13 |
| §15 | Register name embedding | 57.47% | 8/13 |
| §16 | Static analysis flags (cppcheck) | 57.15% | 8/13 |
| §17 | Taint propagation edges | 58.00% | 10/13 |
| §21 | BigVul binary classifier | — | 9/13 |
| §22 | PDG + taint flags combined | — | 9/13 |
| §23 | Sink-node readout + CD cap + residual/LN | 55.40% | 9/13 |
| §24 | PDG + intrinsic-aware sinks (retrain) | 55.00% | 10/13 |
| §25 | PDG slice trained on PrimeVul | 55.56% | 9/13 |
| §26 | Joern PrimeVul — dropped (IKOS pipeline conflict) | — | — |
| §27 | Juliet pretrain (99%+) → Devign BCE FT; multi-feature x(N,3) | 56.12% | **11/13** |
| §28 | Juliet pretrain → Devign RankNet FT; pairwise ranking loss | 44.52% | 11/13 |
| §29 | Juliet-only, no Devign FT — saturates (91–100% on everything) | — | 9/13† |
| §30 | Juliet pos + clean real-C neg (zlib/musl/SQLite); no Devign | — | TBD |

†High cross-run variance (~54–59%) at the ~1,250-sample split scale.

Full experiment notes: [`docs/ir-embed.md`](https://github.com/johwes/llvm-ir-vuln-gnn/blob/main/docs/ir-embed.md)

## Context enrichment for harness generation

`slice_context.py` converts the PDG slice graph into a structured vulnerability
specification for injection into LLM harness generation prompts:

```python
from preprocess_slice_pdg import ir_to_graph_slice_pdg
from slice_context import summarize_slice, format_for_llm

g = ir_to_graph_slice_pdg(open("function.ll").read())
ctx = format_for_llm(summarize_slice(g, fn_name="process_packet"), score=0.91)
# → inject ctx into your LLM harness generation prompt
```

Output example:
```
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

See [`docs/oss-fuzz-gen-integration.md`](https://github.com/johwes/llvm-ir-vuln-gnn/blob/main/docs/oss-fuzz-gen-integration.md)
for the full integration guide with oss-fuzz-gen.

## Intended Use

This model is a **zero-cost ranker**, not a hard gate. Recommended pipeline:

```
clang -O0 -fno-inline -S -emit-llvm src/*.c -I include/ -o fn.ll
python scan_ir.py fn.ll --all-functions --threshold 0.5
# → ranked list of functions by vulnerability score
# → feed top-N to LLM for semantic triage
```

Use it to decide *which functions to show an LLM*, not to make final vulnerability
decisions.

## Limitations

- **Topology-only:** node features are opcode categories; identifier names, string
  literals, and type tokens are discarded. Semantic bugs (wrong comparison operator,
  wrong format string, off-by-one in a constant) produce identical IR topology to
  correct code and are undetectable.
- **GNN ceiling ~55–58% on Devign:** 27 experiments across block-level, instruction-level,
  slice variants, Perfograph encoding, VSDG edges, taint propagation, sink-node readout,
  intrinsic-aware sinks, PrimeVul training, and Juliet pretraining all converge in this range.
  §28 (RankNet) trades classification accuracy for ranking quality — Devign acc may read ~50%
  while scarnet ranking improves; evaluate with eval_all_models.py --scarnet. The ceiling is
  Devign's commit-level label noise (~10–20%), not the architecture or dataset quality.
  CodeBERT on source text reaches 63.43% partly because it reads identifier names.
  See `docs/ir-embed.md` for the full analysis.
- **§26 (Joern/PrimeVul) dropped:** Joern achieved ~95% compilation coverage vs ~34%
  for clang, fixing the PrimeVul attrition problem. Dropped because Joern requires a
  Java runtime and a separate analysis pass that conflicts with the SCAR pipeline
  (IKOS static analysis already uses LLVM bitcode; all experiments stay within that
  toolchain).
- **Real-world ranking is more reliable than Devign accuracy:** §12 scores 56.48% on
  Devign but 84.6% recall on scarnet. The ranking signal is real; the Devign binary
  accuracy is noise-floored.
- **Devign distribution:** trained on C from four large open-source projects; may not
  generalise well to embedded, kernel, or heavily macro-expanded code.

## Repository & Reproducibility

**[johwes/llvm-ir-vuln-gnn](https://github.com/johwes/llvm-ir-vuln-gnn)**

Key files:
- `train_gnn/train_slice_pdg.py` — §12 training (recommended)
- `train_gnn/train_slice_pdg_v5.py` — §25 PrimeVul training
- `train_gnn/train_slice_pdg_v7.py` — §27 Juliet pretrain → Devign BCE fine-tune
- `train_gnn/train_slice_pdg_v8.py` — §28 Juliet pretrain → Devign RankNet fine-tune
- `train_gnn/preprocess_juliet.py` — Juliet Test Suite preprocessor (§27/§28)
- `train_gnn/preprocess_primevul.py` — PrimeVul dataset preprocessor
- `train_gnn/preprocess_slice_pdg.py` — PDG slice extractor
- `train_gnn/preprocess_slice_pdg_v3.py` — v3 extractor (sink_mask + CD cap)
- `train_gnn/scan_ir.py` — inference CLI (`--context` flag for LLM context)
- `train_gnn/slice_context.py` — PDG slice → LLM prompt context
- `docs/ir-embed.md` — full experiment log §1–§28 (§26 Joern dropped)
- `docs/applications.md` — market gap analysis
- `docs/oss-fuzz-gen-integration.md` — oss-fuzz-gen integration guide

## Citation

```
@misc{scar-ir-gnn-2026,
  title  = {SCAR GNN Defect Detector},
  author = {johnnywesterlund},
  year   = {2026},
  url    = {https://huggingface.co/johnnywesterlund/scar-gnn-defect-detector},
  source = {https://github.com/johwes/llvm-ir-vuln-gnn}
}
```
