#!/usr/bin/env python3
"""
train.py — Train a GNN defect detector on preprocessed Devign graphs.

Requires preprocess.py to have been run first (data/*_graphs.pkl).

Usage:
    python train.py                        # full dataset
    python train.py --epochs 10            # quick sanity check
    python train.py --hidden 32            # smaller model
    python train.py --checkpoint model.pt  # save checkpoint path
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv
from torch_geometric.nn.aggr import AttentionalAggregation

HERE = Path(__file__).parent
DATA = HERE / "data"

N_FEATURES = 45   # must match preprocess.py: 3 structural + 7 opcode + 3 icmp + 2 type + 30 API


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DefectGNN(torch.nn.Module):
    """Two-layer RGCNConv (PDG) → AttentionalAggregation pool → binary classifier.

    Edge types: 0 = CFG (control flow), 1 = DFG (SSA def-use, data flow).
    AttentionalAggregation uses a 2-layer MLP gate: captures non-linear feature
    interactions (e.g. has_strcpy AND no signed-cmp) that a single linear gate
    cannot express.
    """

    def __init__(self, in_features: int = N_FEATURES, hidden: int = 64):
        super().__init__()
        self.conv1 = RGCNConv(in_features, hidden, num_relations=2)
        self.conv2 = RGCNConv(hidden, hidden, num_relations=2)
        gate_nn    = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden // 2, 1),
        )
        self.pool  = AttentionalAggregation(gate_nn=gate_nn)
        self.lin   = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_type, batch):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = self.pool(x, batch)                # (batch_size, hidden)
        return self.lin(x).squeeze(-1)          # (batch_size,)  — raw logits


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graphs(pkl_path: Path) -> list[Data]:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    dataset = []
    for g in raw:
        x          = torch.tensor(g["x"],          dtype=torch.float)
        edge_index = torch.tensor(g["edge_index"],  dtype=torch.long)
        edge_type  = torch.tensor(g["edge_type"],   dtype=torch.long)
        y          = torch.tensor([g["y"]],         dtype=torch.float)

        # Normalise node features (z-score per feature column)
        if x.shape[0] > 1:
            x = (x - x.mean(0)) / (x.std(0) + 1e-8)

        dataset.append(Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y))

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
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int,   default=32)
    ap.add_argument("--checkpoint", type=str,   default="model.pt",
                    help="Where to save the best model")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load data
    for split in ["train", "valid", "test"]:
        if not (DATA / f"{split}_graphs.pkl").exists():
            print(f"Missing data/{split}_graphs.pkl — run preprocess.py first.")
            sys.exit(1)

    print("Loading graphs ...")
    train_data = load_graphs(DATA / "train_graphs.pkl")
    valid_data = load_graphs(DATA / "valid_graphs.pkl")
    test_data  = load_graphs(DATA / "test_graphs.pkl")

    print(f"  train={len(train_data)}  valid={len(valid_data)}  test={len(test_data)}")

    vuln_train  = sum(1 for d in train_data if d.y.item() == 1)
    fixed_train = len(train_data) - vuln_train
    pos_weight  = torch.tensor([fixed_train / vuln_train]).to(device)
    print(f"  train class balance: {vuln_train} vuln / {fixed_train} fixed")
    print(f"  pos_weight: {pos_weight.item():.3f}  (fixed/vuln)\n")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size)

    # Model
    model     = DefectGNN(in_features=N_FEATURES, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: DefectGNN-PDG-v2.1(in={N_FEATURES}, hidden={args.hidden}, relations=2)  "
          f"params={n_params:,}\n")

    # Training loop
    best_val_acc = 0.0
    checkpoint   = Path(args.checkpoint)

    print(f"{'Epoch':>5}  {'Loss':>8}  {'Val Acc':>8}  {'':>6}")
    print("-" * 35)

    for epoch in range(1, args.epochs + 1):
        loss    = train_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight)
        val_acc = evaluate(model, valid_loader, device)
        scheduler.step()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)
            marker = "← best"

        print(f"{epoch:>5}  {loss:>8.4f}  {val_acc:>8.2%}  {marker}")

    # Final test evaluation using best checkpoint
    print(f"\nLoading best checkpoint ({checkpoint}) ...")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.2%}")
    print(f"\nCheckpoint saved to: {checkpoint.resolve()}\n")

    if test_acc >= 0.62:
        print("✓ Matches or exceeds CodeBERT baseline (62%)")
    elif test_acc >= 0.55:
        print("— Below CodeBERT but model is learning; try more epochs or larger hidden")
    else:
        print("✗ Accuracy low — check data quality and class balance")


if __name__ == "__main__":
    main()
