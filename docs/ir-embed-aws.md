# Training the GNN Defect Detector on AWS

AWS setup guide for experiment 4b. The training pipeline lives in
`experiments/ir_embed_demo/train_gnn/`. No ProGraML — graph extraction
is implemented directly in `preprocess.py` using stdlib Python and clang.

For the full step-by-step procedure see `docs/experiments/ir-embed-training.md`.
This doc covers the AWS-specific parts: instance choice, launch, and cost.

---

## Instance selection

| Instance | GPU | VRAM | vCPUs | RAM | On-demand price |
|---|---|---|---|---|---|
| `g4dn.xlarge` | T4 | 16 GB | 4 | 16 GB | ~$0.53/hr |
| `g5.xlarge` | A10G | 24 GB | 4 | 16 GB | ~$1.01/hr |
| `p3.2xlarge` | V100 | 16 GB | 8 | 61 GB | ~$3.06/hr |

**Recommended:** `g4dn.xlarge`. 27K graphs fit on a T4 with room to spare.
Use a **Spot Instance** to cut the price 60–70% — save a checkpoint every
epoch so an interruption is recoverable (train.py does this by default).

---

## Launch

1. AMI: **Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04)** — search
   "Deep Learning" in the AMI catalog. Ships with CUDA 12.x, PyTorch, and
   NVIDIA drivers pre-installed.
2. Storage: **50 GB EBS** (default 8 GB is too small).
3. Security group: inbound SSH (port 22) from your IP only.
4. SSH in: `ssh -i your-key.pem ubuntu@<instance-public-ip>`

---

## Environment setup

```bash
conda activate pytorch

# Confirm GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Install dependencies
pip install gdown torch_geometric

# Match PyG to your CUDA version
TORCH=$(python -c "import torch; print(torch.__version__)")
CUDA=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.',''))")
pip install pyg_lib torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

# clang is pre-installed on the DLAMI — verify
clang --version
```

No LLVM version constraint. `preprocess.py` calls whichever `clang` is
on the PATH; the graph extraction parses IR text directly, so any modern
clang works.

---

## Run the pipeline

```bash
# Copy scripts from your local machine
scp -i your-key.pem -r experiments/ir_embed_demo/train_gnn \
    ubuntu@<ip>:~/train_gnn

ssh -i your-key.pem ubuntu@<ip>
cd ~/train_gnn

# Download Devign, compile to IR, build graphs (~45 min, 8 workers)
python preprocess.py --workers 8

# Train (~2-3 hr on T4)
python train.py --epochs 30 --hidden 64

# Copy checkpoint back and terminate
exit
scp -i your-key.pem ubuntu@<ip>:~/train_gnn/model.pt .
```

Terminate the instance immediately after copying the checkpoint.

---

## Estimated cost

| Step | Time | Cost |
|---|---|---|
| Environment setup | 15 min | ~$0.13 |
| Preprocessing (27K functions, 8 workers) | ~45 min | ~$0.40 |
| Training (30 epochs, T4) | ~2–3 hr | ~$1.10–1.60 |
| **Total** | **~3–4 hr** | **~$2–3** |

50 GB EBS storage: ~$0.20 for the session. Negligible.
