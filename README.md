# IR Structural Embedding — GNN Vulnerability Detection

Research into whether LLVM IR graph structure alone can classify vulnerable C functions,
without access to source identifiers, type names, or string literals.

Full experimental record: **[docs/experiments/ir-embed.md](../../docs/experiments/ir-embed.md)**

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
| `preprocess.py` | Block-level graph extraction | Devign |
| `train.py` | Block-level GCN/RGCN classifier | Devign |
| `preprocess_instr.py` | Instruction-level graph extraction | Devign |
| `train_instr.py` | Instruction-level GNN classifier (§7, §10a) | Devign |
| `preprocess_bigvul.py` | Block-level pair extraction | BigVul |
| `train_triplet.py` | Block-level triplet contrastive (§6) | BigVul |
| `preprocess_instr_bigvul.py` | Instruction-level pair extraction | BigVul |
| `train_instr_triplet.py` | Instruction-level triplet contrastive (§8) | BigVul |
| `train_instr_focal.py` | Focal Contrastive Loss + SAGPooling (§10b) | BigVul |
| `scan_ir.py` | Inference: score a new IR file with `model.pt` | — |
| `debug_predicate.py` | Verify icmp/fcmp predicate extraction via llvmlite | — |

See `train_gnn/MODEL_CARD.md` for the deployed model spec and HuggingFace upload
instructions (`hf_upload.py`).

---

## Results summary

| Experiment | Test Acc | Notes |
|---|---|---|
| Majority-class baseline | 56.6% | |
| Block-level GCNConv | 55.04% | |
| Block-level RGCNConv + DFG | 56.08% | |
| **Block-level best (60ep, h=128)** | **57.84%** | pipeline deliverable (`model.pt`) |
| CodeBERT (Colab T4) | 63.43% | upper bound with source tokens |
| Instruction-level GNN (§7) | 58.00% | first to clear block ceiling |
| BigVul block-level triplet k-NN (§6) | 51.21% | soft collapse, pair-sim ↑ |
| BigVul instr-level triplet k-NN (§8) | 48.39% | soft collapse, pair-sim 0.9984→0.9995 ↑ |
| **§10a: enriched vocab classifier** | pending | icmp/fcmp predicate IDs 80-109 |
| **§10b: FCL + SAGPooling** | pending | focal contrastive + SAGPooling |

**Best deployed model:** `model.pt` (block-level, 57.84%) — used by `scan_ir.py`.
Instruction-level model (58.00%) validated but not deployed (+0.16% does not
justify infrastructure change).

**Why CodeBERT wins by +5.6pp:** it sees identifier names and type tokens.
Our GNN discards all identifiers — only opcode categories survive in LLVM IR graphs.

---

## Reproducing experiments

All commands run from `experiments/ir_embed_demo/train_gnn/`.

**System dependencies** (clang must be in PATH):
```bash
# Debian/Ubuntu
apt install clang-14 llvm-14

# macOS
brew install llvm@14
export PATH="$(brew --prefix llvm@14)/bin:$PATH"
```

`llvmlite` ships with a bundled LLVM 14 for graph parsing. Clang 14 is the safest match; clang 15–17 work in practice but are not tested.

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

### Inference

```bash
# Score a single IR file (requires model.pt)
clang -O0 -fno-inline -S -emit-llvm -o /tmp/target.ll target.c
python scan_ir.py /tmp/target.ll
```
