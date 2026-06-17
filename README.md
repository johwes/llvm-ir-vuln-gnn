# GNN Vulnerability Detector — LLVM IR Experiments

Research into whether LLVM IR graph structure alone can classify vulnerable C functions,
without access to source identifiers, type names, or string literals.

**Models:** [`johnnywesterlund/scar-gnn-defect-detector`](https://huggingface.co/johnnywesterlund/scar-gnn-defect-detector) on Hugging Face  
**Full experimental record:** [docs/ir-embed.md](docs/ir-embed.md) — includes TL;DR and plain-language explanation of results

---

## What's in this directory

### `demo.py` + `ir/` + `samples/`

The original proof-of-concept: normalized opcode-frequency histograms on hand-written
LLVM IR for 5–7 (vuln, fixed) pairs from scarnet. No neural network — pure cosine
distance. Confirmed a per-pair structural signal exists before any training.

```bash
python3 demo.py ir/          # no clang needed, uses hand-written .ll files
./run.sh                     # compiles samples/ with clang -O0
```

### `train_gnn/`

Full GNN training pipeline. Each script corresponds to one experiment series:

| Script | Experiment | Dataset |
|---|---|---|
| `preprocess.py` + `train.py` | Block-level GCN/RGCN classifier (§4) | Devign |
| `preprocess_instr.py` + `train_instr.py` | Instruction-level baseline (§7) | Devign |
| `preprocess_instr_v2.py` + `train_instr_v2.py` | §13 Perfograph + call categories | Devign |
| `preprocess_instr_v3.py` + `train_instr_v3.py` | §14 VSDG memory ordering edges | Devign |
| `preprocess_instr_v4.py` + `train_instr_v4.py` | §15 Register name embedding | Devign |
| `preprocess_instr_v5.py` + `train_instr_v5.py` | §16 Static analysis flags | Devign |
| `preprocess_instr_v6.py` + `train_instr_v6.py` | §17 Taint propagation | Devign |
| `preprocess_slice.py` + `train_slice.py` | DFG backward-slice GNN (§11) | Devign |
| `preprocess_slice_pdg.py` + `train_slice_pdg.py` | PDG slice GNN (§12) | Devign |
| `preprocess_bigvul.py` + `train_triplet.py` | Block-level triplet contrastive (§6) | BigVul |
| `preprocess_instr_bigvul.py` + `train_instr_triplet.py` | Instruction triplet (§8) | BigVul |
| `train_instr_focal.py` | Focal Contrastive Loss + SAGPooling (§10b) | BigVul |
| `preprocess_bigvul_cls.py` + `train_bigvul_cls.py` | §21 Standard binary classifier; `--combine-devign` for joint training | BigVul (+ Devign) |
| `preprocess_slice_pdg_v2.py` + `train_slice_pdg_v2.py` | §22 PDG slice + taint flags (Pattern A/B on PDG nodes) | Devign |
| `eval_all_models.py` | Score all checkpoints against any IR corpus | — |
| `scan_ir.py` | Score a single IR file with one model | — |

See `train_gnn/MODEL_CARD.md` for the deployed model spec and HuggingFace upload
instructions (`hf_upload.py`).

---

## Results summary

### Devign benchmark (20 experiments)

| Experiment | Test Acc | Notes |
|---|---|---|
| Majority-class baseline | 56.6% | |
| Block-level best — §4d (`model.pt`) | 57.84% | pipeline deliverable |
| Instruction-level baseline — §7 (`model_instr.pt`) | 58.00% | first to clear block ceiling |
| §11 DFG slice (`model_slice.pt`) | 56.64% | |
| §12 PDG slice (`model_slice_pdg.pt`) | 56.48% | best real-world result (see below) |
| §13 Perfograph + call categories (`model_instr_v2.pt`) | **58.75%** | best Devign accuracy |
| §14 VSDG memory ordering edges | 57.47% | |
| §15 Register name embedding | 57.47% | best zlib CVE rank (see below) |
| §16 Static analysis flags | 57.15% | |
| §17 Taint propagation | 58.00% | |
| §21 BigVul binary (`model_bigvul_cls.pt`) | — | CVE-level labels; balanced acc 61% on BigVul test |
| §21 BigVul+Devign combined (`model_bigvul_combined.pt`) | — | no gain vs BigVul-only on scarnet |
| §22 PDG + taint flags (`model_slice_pdg_v2.pt`) | 56.75% | regressed to 9/13 scarnet — semantic heuristics hurt OOD |
| CodeBERT (reference) | 63.43% | reads C source with identifier names |

**Ceiling: ~57–58%** across all architectures. The gap to CodeBERT is structural — LLVM IR discards variable names and string literals before the model sees anything. See [docs/ir-embed.md](docs/ir-embed.md) for the full analysis.

### Real-world evaluation

| Corpus | Metric | Best result |
|---|---|---|
| scarnet (19 functions, 13 known-vulnerable) | P/R @13 | **84.6%** — §12 PDG slice (11/13 found) |
| zlib v1.2.11 (148 functions, CVE-2018-25032) | Rank of `deflate_stored` | **rank 2/148** — §15; top-10 in 4/9 models |
| §21/§22 (scarnet) | P/R @13 | 9/13 (69.2%) — data quality and semantic heuristics don't transfer |

**Deployed models:** `model.pt`, `model_instr.pt`, `model_slice.pt`, `model_slice_pdg.pt` —
see [`train_gnn/MODEL_CARD.md`](train_gnn/MODEL_CARD.md) and
[HuggingFace](https://huggingface.co/johnnywesterlund/scar-gnn-defect-detector).

---

## Reproducing experiments

All commands run from `train_gnn/`.

**System dependencies** (clang must be in PATH):
```bash
# Debian/Ubuntu
apt install clang llvm

# macOS
brew install llvm
export PATH="$(brew --prefix llvm)/bin:$PATH"
```

`llvmlite` ships with a bundled LLVM. Tested with clang-20 / llvmlite 0.47.0 (LLVM 20).
Clang 14–20 all work for compilation; use `--clang clang-20` with `eval_all_models.py`
if multiple versions are installed.

**Python dependencies:**
```bash
pip install torch torch-geometric llvmlite pandas numpy gdown
```

### Dataset downloads

**Devign** (~60 MB, auto-downloaded by `preprocess.py`):
```bash
# Nothing to do — preprocess.py calls gdown automatically
```

**BigVul** (~10 GB uncompressed, manual download required):
```bash
pip install gdown
gdown 1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X -O data/bigvul.csv
# Result: data/MSR_data_cleaned.csv
```

---

### Devign experiments (block-level classifier)

```bash
# Preprocess — downloads Devign automatically if not present
python preprocess.py

# §4b smoke test
python train.py --epochs 10 --hidden 32

# §4d best (pipeline deliverable)
python train.py --epochs 60 --hidden 128
# Saves: model.pt  (57.84% test accuracy)
```

### Devign experiments (instruction-level classifier)

```bash
# Preprocess — enriched vocab (VOCAB_SIZE=110, icmp/fcmp predicate IDs)
python preprocess_instr.py

# §7 / §10a
python train_instr.py --epochs 30 --hidden 64
# Saves: model_instr.pt  (58.00% test accuracy)
```

### Devign experiments (slice GNN classifiers)

```bash
# §11 — DFG backward-slice GNN
python preprocess_slice.py
python train_slice.py --epochs 30 --hidden 64
# Saves: model_slice.pt  (56.64% test accuracy)

# §12 — PDG slice GNN (DFG + control dependence)
python preprocess_slice_pdg.py
python train_slice_pdg.py --epochs 30 --hidden 64
# Saves: model_slice_pdg.pt  (56.48% test accuracy)
```

### BigVul experiments (block-level contrastive)

```bash
# Preprocess pairs (requires MSR_data_cleaned.csv)
python preprocess_bigvul.py --csv data/MSR_data_cleaned.csv --workers 4

# §6 block-level triplet k-NN
python train_triplet.py --epochs 50 --margin 0.3
# Result: 51.21% k-NN, pair-sim 0.979→0.986 (soft collapse)
```

### BigVul experiments (instruction-level contrastive)

```bash
# Preprocess instruction-level pairs (requires MSR_data_cleaned.csv)
python preprocess_instr_bigvul.py --csv data/MSR_data_cleaned.csv --workers 4

# §8 instruction-level triplet k-NN (baseline)
python train_instr_triplet.py --epochs 50 --margin 0.3
# Result: 48.39% k-NN, pair-sim 0.9984→0.9995 (soft collapse)

# §10b Focal Contrastive Loss + SAGPooling
python train_instr_focal.py --epochs 50 --hidden 64 --tau 0.07 --gamma 2.0
# If training is unstable: --tau 0.1 --gamma 1.0
```

### BigVul experiments (§21 standard binary classifier)

```bash
# Preprocess (requires MSR_data_cleaned.csv)
python preprocess_bigvul_cls.py --csv data/MSR_data_cleaned.csv --workers 4

# §21 BigVul-only classifier
python train_bigvul_cls.py --epochs 30 --hidden 64
# Saves: model_bigvul_cls.pt

# §21 BigVul + Devign combined (requires data/*_instr_v2_graphs.pkl from preprocess_instr_v2.py)
python train_bigvul_cls.py --epochs 30 --hidden 64 --combine-devign
# Saves: model_bigvul_combined.pt
```

### §22 PDG + taint flags

```bash
# Preprocess (adds Pattern A/B taint as second node feature column)
python preprocess_slice_pdg_v2.py --skip-download

# Train (compares against §12 baseline 56.48%)
python train_slice_pdg_v2.py --epochs 30 --hidden 64
# Saves: model_slice_pdg_v2.pt
```

### Inference

```bash
# Score a single IR file (requires model.pt)
clang -O0 -fno-inline -S -emit-llvm -o /tmp/target.ll target.c
python scan_ir.py /tmp/target.ll
```
---

## Evaluating on real codebases

`eval_all_models.py` scores every checkpoint against any directory of `.ll` IR files
and ranks functions by suspicion score. With an answer key it computes P@K and R@K.

### scarnet — planted-bug server (13 known-vulnerable functions)

```bash
# Auto-clones johwes/scarnet, compiles, scores all 9 checkpoints
python eval_all_models.py --scarnet --answer-key scarnet-answer-key.txt

# Reuse previously compiled IR
python eval_all_models.py --ir-dir /tmp/scarnet-ir/ --answer-key scarnet-answer-key.txt
```

### zlib v1.2.11 — real library with a known CVE

```bash
# Compile zlib first (generates zconf.h required before compilation)
git clone --depth 1 --branch v1.2.11 https://github.com/madler/zlib /tmp/zlib-1.2.11
cd /tmp/zlib-1.2.11 && ./configure
mkdir -p /tmp/zlib-ir
for f in *.c; do
    clang-20 -O0 -fno-inline -S -emit-llvm -I. "$f" -o "/tmp/zlib-ir/${f%.c}.ll"
done

# Score all checkpoints; CVE-2018-25032 answer key included in this repo
python eval_all_models.py \
    --ir-dir /tmp/zlib-ir/ \
    --answer-key zlib-v1.2.11-answer-key.txt \
    --top-k 10
```

### Any other codebase

```bash
# Compile to IR, then score
python eval_all_models.py --ir-dir /path/to/ir/ [--answer-key key.txt] [--top-k N]
```
