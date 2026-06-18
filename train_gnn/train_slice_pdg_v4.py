#!/usr/bin/env python3
"""
train_slice_pdg_v4.py — §24: retrain §12 on intrinsic-aware PDG slice graphs.

§12 (preprocess_slice_pdg.py) missed LLVM memory intrinsics as dangerous sinks.
Calls to llvm.memcpy.*, llvm.memmove.*, llvm.memset.* have type-suffixed names
(e.g. llvm.memcpy.p0i8.p0i8.i64) that did not match the bare-name suffix check.
The result: functions whose primary dangerous operation is a memcpy intrinsic
(e.g. Heartbleed's dtls1_process_heartbeat) had the key sink invisible to the
backward slicer and produced degraded training graphs.

Fix (preprocess_slice_pdg.py §24 patch):
  - _is_dangerous() now recognises llvm.<name>.* prefixes
  - _canonical_name() maps intrinsic names back to canonical names
    (e.g. llvm.memcpy.p0i8.p0i8.i64 -> "memcpy") for sink_fn_names lookup
  - Secondary effect: wrapper names like CRYPTO_malloc -> "malloc" are now
    also canonicalized, improving context enrichment descriptions

Workflow:
    python preprocess_slice_pdg.py   # regenerate graphs with intrinsic-aware sinks
    python train_slice_pdg_v4.py     # train on fixed graphs -> model_slice_pdg_v4.pt

Architecture: identical to §12 (SlicePDGGNN, RGCN + AttentionalAggregation).
Only the training data changes — a clean ablation of the preprocessing fix.

Usage:
    python train_slice_pdg_v4.py --epochs 30 --hidden 64
    python train_slice_pdg_v4.py --epochs 60 --hidden 128
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv
from torch_geometric.nn.aggr import AttentionalAggregation

HERE = Path(__file__).parent
DATA = HERE / "data"

VOCAB_SIZE = 110


# ---------------------------------------------------------------------------
# Model — identical to §12 (SlicePDGGNN)
# ---------------------------------------------------------------------------

class SlicePDGGNN(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, embed_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=79)
        self.conv1 = RGCNConv(embed_dim, hidden, num_relations=3)
        self.conv2 = RGCNConv(hidden,    hidden, num_relations=3)
        gate_nn    = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )
        self.pool  = AttentionalAggregation(gate_nn=gate_nn)
        self.lin   = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_type, batch):
        h = self.embed(x.squeeze(-1))
        h = F.relu(self.conv1(h, edge_index, edge_type))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_type))
        h = self.pool(h, batch)
        return self.lin(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graphs(pkl_path: Path) -> list[Data]:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    dataset = []
    for g in raw:
        x          = torch.tensor(g["x"],          dtype=torch.long)
        edge_index = torch.tensor(g["edge_index"],  dtype=torch.long)
        edge_type  = torch.tensor(g["edge_type"],   dtype=torch.long)
        y          = torch.tensor([g["y"]],         dtype=torch.float)
        dataset.append(Data(x=x, edge_index=edge_index,
                            edge_type=edge_type, y=y))
    return dataset


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, pos_weight=None):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        loss   = F.binary_cross_entropy_with_logits(
                     logits, batch.y.squeeze(), pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        preds  = (logits > 0).long()
        labels = batch.y.squeeze().long()
        correct += (preds == labels).sum().item()
        total   += batch.num_graphs
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=30)
    ap.add_argument("--hidden",     type=int,   default=64)
    ap.add_argument("--embed-dim",  type=int,   default=128)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int,   default=32)
    ap.add_argument("--checkpoint", type=str,   default="model_slice_pdg_v4.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print("§24: intrinsic-aware PDG slice retraining")
    print("  Preprocessor fix: llvm.memcpy.* / llvm.memmove.* / llvm.memset.*"
          " now recognised as dangerous sinks\n")

    for split in ["train", "valid", "test"]:
        if not (DATA / f"{split}_slice_pdg_graphs.pkl").exists():
            print(f"Missing data/{split}_slice_pdg_graphs.pkl "
                  f"-- run preprocess_slice_pdg.py first.")
            sys.exit(1)

    print("Loading PDG slice graphs ...")
    train_data = load_graphs(DATA / "train_slice_pdg_graphs.pkl")
    valid_data = load_graphs(DATA / "valid_slice_pdg_graphs.pkl")
    test_data  = load_graphs(DATA / "test_slice_pdg_graphs.pkl")
    print(f"  train={len(train_data)}  valid={len(valid_data)}  test={len(test_data)}")

    node_counts = [d.x.shape[0] for d in train_data]
    import statistics
    print(f"  Train slice sizes: mean={statistics.mean(node_counts):.0f}  "
          f"median={statistics.median(node_counts):.0f}  max={max(node_counts)}")

    vuln_train  = sum(1 for d in train_data if d.y.item() == 1)
    fixed_train = len(train_data) - vuln_train
    pos_weight  = torch.tensor([fixed_train / vuln_train]).to(device)
    print(f"  train class balance: {vuln_train} vuln / {fixed_train} fixed")
    print(f"  pos_weight: {pos_weight.item():.3f}\n")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size)

    model     = SlicePDGGNN(VOCAB_SIZE, args.embed_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SlicePDGGNN(vocab={VOCAB_SIZE}, embed={args.embed_dim}, "
          f"hidden={args.hidden}, relations=3)  params={n_params:,}\n")

    best_val_acc = 0.0
    checkpoint   = Path(args.checkpoint)

    print(f"{'Epoch':>5}  {'Loss':>8}  {'Val Acc':>8}  {'':>6}")
    print("-" * 35)

    for epoch in range(1, args.epochs + 1):
        loss    = train_epoch(model, train_loader, optimizer, device, pos_weight)
        val_acc = evaluate(model, valid_loader, device)
        scheduler.step()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)
            marker = "<- best"

        print(f"{epoch:>5}  {loss:>8.4f}  {val_acc:>8.2%}  {marker}")

    print(f"\nLoading best checkpoint ({checkpoint}) ...")
    model.load_state_dict(torch.load(checkpoint, map_location=device,
                                     weights_only=True))
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.2%}")
    print(f"\nCheckpoint saved to: {checkpoint.resolve()}\n")

    print("--- Results ---")
    print(f"  §12 PDG slice (no intrinsic recognition):                   56.48%")
    print(f"  §24 PDG slice + intrinsic-aware sinks (this run):           {test_acc:.2%}")

    delta = test_acc - 0.5648
    if delta > 0.005:
        print(f"\n  +{delta:.2%} vs §12 — intrinsic recognition improved training signal")
    elif delta > 0:
        print(f"\n  +{delta:.2%} vs §12 — marginal improvement; "
              "context enrichment benefit is the main gain")
    else:
        print(f"\n  {delta:.2%} vs §12 — no Devign improvement; "
              "intrinsic fix primarily benefits context enrichment (harness hints)")


if __name__ == "__main__":
    main()
