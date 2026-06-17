#!/usr/bin/env python3
"""
train_slice_pdg_v2.py — §22 PDG slice GNN with taint features.

Extends §12 (train_slice_pdg.py) by adding a taint feature column to each
node, computed by preprocess_slice_pdg_v2.py.

The PDG slice already selects the right subgraph (dangerous sink + its
data/control dependencies). Taint flags add an explicit per-node annotation:
  - 1.0 at call nodes identified as Pattern A (unguarded dangerous call) or
    Pattern B (unchecked alloc/IO/network return value)
  - 0.5 / 0.25 at downstream nodes that receive the tainted value via DFG

Hypothesis: combining the focused PDG structure with explicit taint annotations
increases the score separation between vulnerable and clean functions, even if
the hit count at P@K stays at 11/13. Better score separation → more reliable
threshold for the SCAR pipeline filter.

Architecture:
  Embedding(110, embed_dim) for opcode
  Concatenate taint float → in_dim = embed_dim + 1
  RGCNConv(in_dim,  hidden, 3 relations)
  RGCNConv(hidden,  hidden, 3 relations)
  AttentionalAggregation → Linear(hidden, 1)

Node features: x is (N, 2) float32
  x[:, 0] = opcode_id  (cast to long for nn.Embedding)
  x[:, 1] = taint_value (0.0–1.0)

Usage:
    python preprocess_slice_pdg_v2.py --skip-download   # first
    python train_slice_pdg_v2.py --epochs 30 --hidden 64
    python train_slice_pdg_v2.py --epochs 60 --hidden 128
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
# Model
# ---------------------------------------------------------------------------

class SlicePDGGNNv2(nn.Module):
    """PDG slice GNN with taint feature concatenated after opcode embedding."""

    def __init__(self, vocab: int = VOCAB_SIZE, embed_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.embed  = nn.Embedding(vocab, embed_dim, padding_idx=79)
        in_dim      = embed_dim + 1          # opcode embedding + taint float
        self.conv1  = RGCNConv(in_dim, hidden, num_relations=3)
        self.conv2  = RGCNConv(hidden, hidden, num_relations=3)
        gate_nn     = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )
        self.pool   = AttentionalAggregation(gate_nn=gate_nn)
        self.lin    = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_type, batch):
        opcode = self.embed(x[:, 0].long())   # (N, embed_dim)
        taint  = x[:, 1:2]                    # (N, 1) float
        h = torch.cat([opcode, taint], dim=1) # (N, embed_dim + 1)
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
        x          = torch.tensor(g["x"],         dtype=torch.float)  # (N, 2) float32
        edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
        edge_type  = torch.tensor(g["edge_type"],  dtype=torch.long)
        y          = torch.tensor([g["y"]],        dtype=torch.float)
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
    ap.add_argument("--checkpoint", type=str,   default="model_slice_pdg_v2.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    for split in ["train", "valid", "test"]:
        if not (DATA / f"{split}_slice_pdg_v2_graphs.pkl").exists():
            print(f"Missing data/{split}_slice_pdg_v2_graphs.pkl "
                  f"-- run preprocess_slice_pdg_v2.py first.")
            sys.exit(1)

    print("Loading PDG v2 slice graphs ...")
    train_data = load_graphs(DATA / "train_slice_pdg_v2_graphs.pkl")
    valid_data = load_graphs(DATA / "valid_slice_pdg_v2_graphs.pkl")
    test_data  = load_graphs(DATA / "test_slice_pdg_v2_graphs.pkl")
    print(f"  train={len(train_data)}  valid={len(valid_data)}  test={len(test_data)}")

    node_counts = [d.x.shape[0] for d in train_data]
    n_tainted   = sum(1 for d in train_data if d.x[:, 1].max().item() > 0)
    import statistics
    print(f"  Train slice sizes: mean={statistics.mean(node_counts):.0f}  "
          f"median={statistics.median(node_counts):.0f}  max={max(node_counts)}")
    print(f"  Taint coverage: {n_tainted}/{len(train_data)} graphs "
          f"({100*n_tainted/len(train_data):.0f}%) have ≥1 tainted node")

    vuln_train  = sum(1 for d in train_data if d.y.item() == 1)
    fixed_train = len(train_data) - vuln_train
    pos_weight  = torch.tensor([fixed_train / vuln_train]).to(device)
    print(f"  train class balance: {vuln_train} vuln / {fixed_train} fixed")
    print(f"  pos_weight: {pos_weight.item():.3f}\n")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size)

    model     = SlicePDGGNNv2(VOCAB_SIZE, args.embed_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SlicePDGGNNv2(vocab={VOCAB_SIZE}, embed={args.embed_dim}, "
          f"hidden={args.hidden}, relations=3, in_dim={args.embed_dim+1})  "
          f"params={n_params:,}\n")

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
    print(f"  §12 PDG slice (no taint, 30ep h=64):                      56.48%")
    print(f"  §22 PDG + taint flags (this run):                          {test_acc:.2%}")
    print()
    print("Run eval_all_models.py --scarnet to compare real-world scarnet performance.")
    print("Key metric: score gap between true and false positives, not just hit count.")


if __name__ == "__main__":
    main()
