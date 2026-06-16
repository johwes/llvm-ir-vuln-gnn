# Training the Defect-Detection GNN — Step-by-Step

Practical guide to go from zero to a trained GNN on the Devign dataset.
All scripts live in `train_gnn/`.

---

## What you need

| Tool | Minimum version | Notes |
|---|---|---|
| Python | 3.10+ | |
| clang | any (14 recommended) | Must be on `PATH` |
| pip | recent | |
| ~3 GB disk | | dataset + IR files + model |

For laptop testing: CPU is fine — use `--subset` to keep runtimes short.
For full training: see the AWS section at the bottom.

---

## Directory layout

```
train_gnn/
├── requirements.txt        ← Python dependencies
├── preprocess.py           ← download + compile + build graphs (block-level)
├── train.py                ← train block-level GNN, save model.pt
├── preprocess_instr.py     ← instruction-level graph extractor
├── train_instr.py          ← train instruction-level GNN, save model_instr.pt
├── preprocess_slice.py     ← DFG backward slice extractor (§11)
├── train_slice.py          ← train slice GNN, save model_slice.pt
├── preprocess_slice_pdg.py ← PDG slice extractor (§12)
├── train_slice_pdg.py      ← train PDG slice GNN, save model_slice_pdg.pt
├── scan_ir.py              ← score a function with a trained model
└── data/                   ← created by preprocess.py (not in git)
    ├── devign.json         ← raw Devign download
    ├── train.jsonl         ← 80% split
    ├── valid.jsonl         ← 10% split
    ├── test.jsonl          ← 10% split
    ├── train_graphs.pkl    ← compiled + parsed graphs
    ├── valid_graphs.pkl
    └── test_graphs.pkl
```

---

## Step 1 — Install dependencies

```bash
cd train_gnn

# CPU-only install (laptop):
pip install gdown
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric

# GPU install (AWS — replace cu126 with your CUDA version):
pip install gdown torch torch_geometric
pip install pyg_lib torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cu126.html
```

Verify:
```bash
python -c "import torch; import torch_geometric; print('ok')"
```

### Optional but strongly recommended: install project headers

Devign functions use FFmpeg and LibTIFF types extensively. Without the
real headers, `AVCodecContext`, `TIFF *`, etc. can't be resolved and
attrition is ~90-95%. With them, it drops to ~40-60%.

```bash
# Fedora / RHEL:
sudo dnf install ffmpeg-free-devel libtiff-devel

# Ubuntu / Debian:
sudo apt install libavcodec-dev libavutil-dev libavformat-dev libtiff-dev

# macOS (Homebrew):
brew install ffmpeg libtiff
```

`preprocess.py` auto-detects these headers and uses them automatically —
no flags needed. It prints which headers it found at startup.

---

## Step 2 — Download the Devign dataset

The dataset is hosted on Google Drive via the CodeXGLUE benchmark.
`preprocess.py` downloads it automatically using `gdown`:

```bash
python preprocess.py --subset 500   # laptop: 500 examples → ~240 graphs, ~2 min
python preprocess.py                # full: 27K examples → ~10K graphs, ~30-60 min
```

What it does:
1. Downloads `devign.json` (~50 MB) from Google Drive to `data/`
2. Splits 80/10/10 into `train.jsonl`, `valid.jsonl`, `test.jsonl`
3. For each C function: prepends common headers, calls `clang -O0 -S -emit-llvm`
4. Parses the LLVM IR to extract a CFG graph with node features
5. Saves pickled graph lists to `data/*_graphs.pkl`

**Expected compile attrition:** Devign functions reference project-specific
types (AVCodecContext, sk_buff, CPUState, etc.). The stub injector handles
most of these: it stubs unknown types as padded structs, then injects
missing member names as int fields when clang reports "no member named X".
With `-ferror-limit=0` all missing members are reported in one pass, so
most functions succeed in 2–3 iterations. Expect **30–60% attrition**
with member injection; without it the figure was ~95%.

**Attrition varies by project:** Devign mixes FFmpeg, QEMU, Linux kernel,
and LibTIFF. Functions with deep pointer chains (`avctx->priv_data->field`)
or complex macro dependencies still fail; simple to moderate struct access
now compiles. The `--subset` flag uses a random balanced sample so you see
representative attrition rather than a worst-case FFmpeg-only sample.

**Installing FFmpeg/LibTIFF dev headers** (Step 1) brings two additional
benefits: correct macro values (AV_CODEC_ID_*, AV_PIX_FMT_*, etc.) and
correct function signatures for library calls. The installed dev headers
must match the FFmpeg version used in Devign (circa 2016); newer versions
may have removed fields, causing "no member named" on real struct definitions
which the stub injector cannot fix. Using stubs + member injection sidesteps
the version mismatch entirely.

**Parallel workers:** default is 4. Increase on AWS:
```bash
python preprocess.py --workers 16
```

---

## Step 3 — What the graph looks like

Each basic block becomes a node with 11 features:

| Feature | Description |
|---|---|
| `n_instructions` | Number of IR instructions in this block |
| `out_degree` | Number of outgoing CFG edges |
| `in_degree` | Number of incoming CFG edges |
| `has_call` | 1 if block contains a function call |
| `has_store` | 1 if block writes to memory |
| `has_load` | 1 if block reads from memory |
| `has_icmp` | 1 if block contains a comparison |
| `has_alloca` | 1 if block allocates stack memory |
| `has_getelementptr` | 1 if block does pointer arithmetic |
| `has_ret` | 1 if block is a return block |
| `has_br` | 1 if block ends with a branch |

Edges connect basic blocks that can transfer control flow.
The GNN propagates these features through the graph before classifying.

---

## Step 4 — Train on your laptop (CPU)

```bash
python train.py --epochs 10 --hidden 32
```

With `--subset 500` from the preprocess step this runs in a few minutes
and is enough to confirm the pipeline works. Don't expect high accuracy
at this scale — you're checking the code runs cleanly.

Output:
```
Device: cpu
Loading graphs ...
  train=400  valid=50  test=50
  train class balance: 200 vuln / 200 fixed

Model: DefectGNN(in=11, hidden=32)  params=2,305

Epoch    Loss   Val Acc
----------------------------
    1  0.6931    50.00%
    5  0.6712    54.00%  ← best
   10  0.6604    56.00%  ← best
```

At subset scale (400 training graphs) results are noisy — this is a pipeline smoke test,
not a meaningful accuracy measurement.

---

## Step 5 — Full training on AWS

See `docs/experiments/ir-embed-aws.md` for instance selection and setup.
Short version:

- **Instance:** `g4dn.xlarge` (~$0.53/hr, T4 GPU)
- **AMI:** Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04)
- **Storage:** 50 GB EBS

Once the instance is running:

```bash
# Copy scripts
scp -i your-key.pem -r experiments/ir_embed_demo/train_gnn \
    ubuntu@<ip>:~/train_gnn

ssh -i your-key.pem ubuntu@<ip>

# On the instance
conda activate pytorch
cd train_gnn
pip install gdown torch_geometric

python preprocess.py --workers 8      # ~45 min
python train.py --epochs 30           # ~2-3 hr on T4

# Copy checkpoint back
exit
scp -i your-key.pem ubuntu@<ip>:~/train_gnn/model.pt .

# Terminate immediately to stop billing
```

Full training hyperparameters (block-level, best published result 57.84%):
```bash
python train.py --epochs 60 --hidden 128 --checkpoint model.pt
```

Instruction-level (best published result 58.00% — peaks at epoch ~17, larger runs overfit):
```bash
python preprocess_instr.py
python train_instr.py --epochs 30 --hidden 64 --checkpoint model_instr.pt
```

Slice variants (both ~56.5%):
```bash
python preprocess_slice.py && python train_slice.py --epochs 30 --hidden 64
python preprocess_slice_pdg.py && python train_slice_pdg.py --epochs 30 --hidden 64
```

---

## Expected results

| Scale | Accuracy target | Notes |
|---|---|---|
| 500 samples, 10 epochs (laptop) | 50–56% | Sanity check only |
| Full Devign, 30–60 epochs (AWS) | 55–58% | Empirical ceiling with current features |

**The structural ceiling on Devign is ~57–58%** across all 12 experiments run in this
project (block-level, instruction-level, slice-based, various loss functions). The
majority-class baseline is 56.6%, so meaningful learning does occur — but the gap to
CodeBERT (63.4%) is a representation problem, not an architecture problem. CodeBERT
reads identifier names and string literals from source; our GNN sees only opcode
categories. See `docs/ir-embed.md` § "Future Directions: Feature Extraction Improvements"
for the ranked list of what would actually move this number.

---

## Troubleshooting

**`clang: command not found`**
Install clang: `sudo apt install clang` (Linux) or via Homebrew on Mac.
On the scar-agent container clang 14 is pre-installed.

**`gdown` fails / Google Drive quota exceeded**
Try again later, or download the file manually from:
`https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF`
and place it at `data/devign.json`.

**`torch_geometric` import error**
PyG version must match your PyTorch version. Check:
`python -c "import torch; print(torch.__version__)"` and reinstall PyG
against that version.

**Very high attrition (>70%) even with random sampling**
With both fixes applied (stub injection + `#define static`/`inline`),
expect ~52% attrition (48% survival) and `graphed` matching `compiled`.
If you're seeing higher attrition, check:

- **`graphed` much lower than `compiled`**: means the `#define static`
  fix is not active. Verify you have the latest `preprocess.py` — the
  end of `_PREAMBLE_STATIC` should contain `#define static` and
  `#define inline`. Without this, clang silently omits `static`/`inline`
  functions from the `.ll` output (no `define` lines, empty IR).
- **`compiled` itself low**: remaining compile failures are from deep
  pointer chains (`avctx->priv_data->field`) and complex macro
  dependencies. These are expected; the 48% that do compile is enough
  for ~8K training graphs on the full dataset.

If attrition remains above 70%, run the full 27K dataset — at 48%
survival you get ~8K training graphs, well above the minimum for
meaningful GNN training.

**Three paths for near-zero attrition:**

*Option A — Use the Devign source code directly (no compilation).*
The 4a experiment (CodeBERT/UniXcoder) trains on raw C source tokens and
achieves 62–69% on the Devign test set. No clang, no IR, no attrition.
Friction is minimal: `pip install transformers` and run the CodeXGLUE
`run.py` script. This is the right path for getting a trained classifier
on Devign.

*Option B — Build the projects with real headers.*
Clone FFmpeg, QEMU, Linux, and LibTIFF, build each with
`-flto -fembed-bitcode` to emit whole-program bitcode, then use
`llvm-extract --func=NAME` to pull per-function IR. This is the approach
ProGraML used. Attrition drops to near zero but setup takes several hours.

*Option C — Use a standalone-compilable dataset.*
The NIST Juliet Test Suite (~28K C/C++ cases, CWE-labeled, no external
headers required) compiles standalone. Attrition with the stub injector
will be <10%. Trade-off: synthetic code, limited generalization to
real-world vulnerability patterns.

For SCAR's own integration target (the `ir-embed-scan` Tekton task), the
attrition problem does not exist — the full build system is available and
functions are compiled in their proper project context.

**Training loss stuck at ~0.69 (log(2))**
The model is predicting 50/50. Try:
- More epochs
- Lower learning rate (`--lr 1e-4`)
- Larger hidden dimension (`--hidden 128`)

---

## Retraining cookbook — implementing the Tier 1 improvements

The feature extraction improvements ranked in `docs/ir-embed.md` § "Future Directions:
Feature Extraction Improvements" each have a specific touch point. This section maps
improvement → files to change → how to evaluate.

---

### 1. Perfograph constant encoding

**Touch point:** `preprocess.py` and `preprocess_instr.py`, wherever constants are
currently encoded as a binary flag.

In `preprocess.py` (block-level), find the node feature vector construction and replace
any `has_constant` or similar binary flag with:

```python
import math
def encode_constant(val):
    return math.copysign(math.log2(abs(val) + 1), val) if val != 0 else 0.0
```

In `preprocess_instr.py` (instruction-level), the constant node currently gets opcode
ID 76/77 (constant int/fp). Augment the node feature with the encoded value as a
second channel alongside the opcode index.

**Evaluate:** re-run `preprocess.py` → `train.py --epochs 60 --hidden 128`. Compare
test accuracy to the 57.84% baseline.

---

### 2. Categorical call target mapping

**Touch point:** `preprocess.py` (`has_call` flag) and `preprocess_instr.py` (mock node
naming in Pass 3).

Replace the single `has_call` binary with 6 binary flags, one per category:

```python
_CALL_BUCKETS = {
    "alloc":   {"malloc","calloc","realloc","kmalloc","kzalloc","av_malloc","g_malloc","g_new"},
    "copy":    {"memcpy","memmove","memset"},
    "string":  {"strcpy","strncpy","strcat","strncat","sprintf","snprintf","gets","fgets","scanf"},
    "file_io": {"fopen","fread","fwrite","fclose","read","write","open"},
    "network": {"recv","send","accept","connect","recvfrom","sendto"},
}
# anything else → "internal" (no flag set)
```

In `preprocess_instr.py` the mock node already stores the call target name — extend
`_instr_node_id()` to return a bucket-specific opcode ID for call instructions targeting
known-dangerous functions rather than a single generic `call` ID.

**Evaluate:** re-run preprocess + train. Check whether Scarnet false negatives
(`handle_set`, `handle_del`) improve in `eval_scarnet.sh`.

---

### 3. IR2Vec vocabulary replacement

**Touch point:** `preprocess_instr.py` (node feature extraction) and `train_instr.py`
(replace `nn.Embedding` with pre-computed float vectors).

**Step 1** — generate IR2Vec embeddings for each `.ll` file:

```bash
# Requires LLVM 17+ with IR2Vec analysis pass
opt -passes=ir2vec-vocab -ir2vec-vocab-file=vocab.bin input.ll -o /dev/null
```

IR2Vec outputs a fixed-size vector per instruction. Export to numpy:

```python
# Use llvmlite or llvm-project Python bindings to extract per-instruction embeddings
# and save as a dict {function_name: {instr_ptr: np.ndarray(300,)}}
```

**Step 2** — in `preprocess_instr.py`, replace the opcode integer index with the
300-dim IR2Vec float vector as the node feature.

**Step 3** — in `train_instr.py`, replace `nn.Embedding(VOCAB_SIZE, 128)` with a
`nn.Linear(300, 128)` projection layer as the first step.

**Evaluate:** re-run preprocess_instr + train_instr. Compare to the 58.00% instruction-
level baseline.

---

### Retraining from SCAR accepted patches (self-improving corpus)

Each SCAR accepted patch on any target is a labelled `(vulnerable IR, fixed IR)` pair
at zero marginal cost. To incorporate new patches into the training data:

1. For each `accepted` entry in `scar-results.json`:
   - The original source file (before patch) is the vulnerable version.
   - Apply the patch with `patch -o fixed.c original.c patch.diff`.
   - Compile both: `clang -O0 -fno-inline -S -emit-llvm`.
   - Extract the enclosing function from each `.ll`.

2. Run `preprocess.py` (or `preprocess_instr.py`) on the new IR files to produce graph
   objects in the same pickle format as the Devign data.

3. Append to `data/train_graphs.pkl`:
   ```python
   import pickle
   with open("data/train_graphs.pkl", "rb") as f:
       existing = pickle.load(f)
   existing.extend(new_scar_graphs)
   with open("data/train_graphs.pkl", "wb") as f:
       pickle.dump(existing, f)
   ```

4. Re-run `train.py` (or `train_instr.py`) with the same hyperparameters.
   The model specialises to the vulnerability patterns SCAR encounters in practice.

**Trigger:** after every 50–100 new accepted patches, or when Scarnet evaluation
(`eval_scarnet.sh`) shows regression on previously-detected functions.
