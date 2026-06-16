#!/usr/bin/env python3
"""
train_instr_v3.py — §14 instruction-level GNN: VSDG memory ordering edges.

Requires preprocess_instr_v3.py to have been run first (data/*_instr_v3_graphs.pkl).

Architecture (identical to §13 except num_relations 3 → 4):
  nn.Embedding(111, 128)           # opcode vocab -> dense embedding
  cat(embedding, const_magnitude)  # (N, 129)
  RGCNConv(129 -> 64, 4 rels)     # 0=CFG, 1=DFG, 2=Global, 3=State
  RGCNConv(64  -> 64, 4 rels)
  AttentionalAggregation
  Linear(64 -> 1)

Changes vs train_instr_v2.py (§13):
  - num_relations: 3 -> 4 (adds State relation for load/store ordering edges)
  - Data files: *_instr_v3_graphs.pkl
  - Checkpoint default: model_instr_v3.pt

Usage:
    python train_instr_v3.py --epochs 30 --hidden 64
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

VOCAB_SIZE = 111


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class InstructionGNN(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, embed_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=79)
        self.conv1 = RGCNConv(embed_dim + 1, hidden, num_relations=4)
        self.conv2 = RGCNConv(hidden,        hidden, num_relations=4)
        gate_nn    = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )
        self.pool  = AttentionalAggregation(gate_nn=gate_nn)
        self.lin   = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_type, batch):
        opcode_ids = x[:, 0].long()
        const_mag  = x[:, 1].unsqueeze(-1)
        h = torch.cat([self.embed(opcode_ids), const_mag], dim=-1)
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
        x          = torch.nan_to_num(
                         torch.tensor(g["x"], dtype=torch.float),
                         nan=0.0, posinf=0.0, neginf=0.0)
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
    ap.add_argument("--checkpoint", type=str,   default="model_instr_v3.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    for split in ["train", "valid", "test"]:
        if not (DATA / f"{split}_instr_v3_graphs.pkl").exists():
            print(f"Missing data/{split}_instr_v3_graphs.pkl -- run preprocess_instr_v3.py first.")
            sys.exit(1)

    print("Loading graphs ...")
    train_data = load_graphs(DATA / "train_instr_v3_graphs.pkl")
    valid_data = load_graphs(DATA / "valid_instr_v3_graphs.pkl")
    test_data  = load_graphs(DATA / "test_instr_v3_graphs.pkl")
    print(f"  train={len(train_data)}  valid={len(valid_data)}  test={len(test_data)}")

    vuln_train  = sum(1 for d in train_data if d.y.item() == 1)
    fixed_train = len(train_data) - vuln_train
    pos_weight  = torch.tensor([fixed_train / vuln_train]).to(device)
    print(f"  train class balance: {vuln_train} vuln / {fixed_train} fixed")
    print(f"  pos_weight: {pos_weight.item():.3f}\n")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size)

    model     = InstructionGNN(VOCAB_SIZE, args.embed_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: InstructionGNN(vocab={VOCAB_SIZE}, embed={args.embed_dim}, "
          f"hidden={args.hidden}, relations=4)  params={n_params:,}\n")

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

    print("--- Comparison ---")
    print(f"  §7  Instruction-level GNN baseline:          58.00%")
    print(f"  §13 + Perfograph + call categorization:      58.75% (best single run)")
    print(f"  §14 + VSDG memory ordering edges (this run): {test_acc:.2%}")

    delta = test_acc - 0.5875
    if delta > 0.005:
        print(f"\n  +{delta:.2%} over §13 — memory ordering edges add signal")
    elif delta > -0.005:
        print(f"\n  Within noise of §13 — state edges have no net effect at this scale")
    else:
        print(f"\n  Below §13 — added edges may introduce noise; check edge count stats")


if __name__ == "__main__":
    main()
