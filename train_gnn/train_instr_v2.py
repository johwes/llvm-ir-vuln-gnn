#!/usr/bin/env python3
"""
train_instr_v2.py — §13 instruction-level GNN: Perfograph + categorical call targets.

Requires preprocess_instr_v2.py to have been run first (data/*_instr_v2_graphs.pkl).

Architecture:
  nn.Embedding(111, 128)           # opcode vocab (111 IDs) -> dense embedding
  cat(embedding, const_magnitude)  # (N, 129): append Perfograph-encoded constant magnitude
  RGCNConv(129 -> 64, 3 rels)     # relations: 0=CFG, 1=DFG, 2=Global
  RGCNConv(64 -> 64, 3 rels)
  AttentionalAggregation
  Linear(64 -> 1)

Changes vs train_instr.py (§7 baseline):
  - Node features: (N, 2) float32 [opcode_id, const_magnitude] instead of (N, 1) int64
  - VOCAB_SIZE 111: adds IDX_MOCK_ALLOC/COPY/STRING/FILEIO/NETWORK (106-110)
  - conv1 in_features: embed_dim + 1 = 129 (appended constant magnitude channel)

Usage:
    python train_instr_v2.py --epochs 30 --hidden 64
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

VOCAB_SIZE = 111  # 0-105 original + 106-110 call category mock IDs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class InstructionGNN(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, embed_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=79)
        # conv1 input: embed_dim + 1 (the appended Perfograph constant magnitude channel)
        self.conv1 = RGCNConv(embed_dim + 1, hidden, num_relations=3)
        self.conv2 = RGCNConv(hidden,        hidden, num_relations=3)
        gate_nn    = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )
        self.pool  = AttentionalAggregation(gate_nn=gate_nn)
        self.lin   = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_type, batch):
        # x: (N, 2) float32 — col 0 = opcode_id, col 1 = Perfograph constant magnitude
        opcode_ids = x[:, 0].long()
        const_mag  = x[:, 1].unsqueeze(-1)                     # (N, 1)
        h = torch.cat([self.embed(opcode_ids), const_mag], dim=-1)  # (N, embed_dim+1)
        h = F.relu(self.conv1(h, edge_index, edge_type))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_type))
        h = self.pool(h, batch)                                 # (B, hidden)
        return self.lin(h).squeeze(-1)                          # (B,) raw logits


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
    ap.add_argument("--checkpoint", type=str,   default="model_instr_v2.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    for split in ["train", "valid", "test"]:
        if not (DATA / f"{split}_instr_v2_graphs.pkl").exists():
            print(f"Missing data/{split}_instr_v2_graphs.pkl -- run preprocess_instr.py first.")
            sys.exit(1)

    print("Loading graphs ...")
    train_data = load_graphs(DATA / "train_instr_v2_graphs.pkl")
    valid_data = load_graphs(DATA / "valid_instr_v2_graphs.pkl")
    test_data  = load_graphs(DATA / "test_instr_v2_graphs.pkl")
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

    print("--- Comparison ---")
    print(f"  Block-level classifier best (4d, 60ep h=128):  57.84%")
    print(f"  CodeBERT fine-tuned (4a):                      63.43%")
    print(f"  Instruction-level GNN (this run):               {test_acc:.2%}")

    if test_acc >= 0.62:
        print("\n  Instruction-level breaks the block-level ceiling (57.84%) and "
              "reaches CodeBERT territory -- instruction-level micro-topology matters")
    elif test_acc >= 0.58:
        print("\n  Marginal improvement over block-level; "
              "instruction resolution helps but embedding may need tuning")
    else:
        print("\n  Below block-level baseline; check data quality and "
              "consider longer training or larger embed_dim")


if __name__ == "__main__":
    main()
