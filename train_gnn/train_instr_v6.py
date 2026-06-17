#!/usr/bin/env python3
"""
train_instr_v6.py — §17 instruction-level GNN: taint propagation + extended patterns.

Requires preprocess_instr_v6.py to have been run first (data/*_instr_v6_graphs.pkl).

Architecture (same shape as §16 — only the x[:,2] semantics change):
  nn.Embedding(111, 128)              # opcode vocab -> dense embedding
  cat(embedding, const_mag, taint)    # (N, 130)  — taint is a float in [0,1]
  RGCNConv(130 -> 64, 3 rels)        # 0=CFG, 1=DFG, 2=Global
  RGCNConv( 64 -> 64, 3 rels)
  AttentionalAggregation
  Linear(64 -> 1)

Changes vs §16 (train_instr_v5.py):
  - x[:,2] is now continuous in [0,1] (taint decay), not binary
    1.0 = source node (Pattern A/B flagged)
    0.5 = 1-hop downstream in DFG
    0.25 = 2-hop
    0.125 = 3-hop
  - Extended Pattern B: covers ALLOC + FILEIO + NETWORK unchecked returns
  - Coverage diagnostic reports both graph-level rate and mean taint value per flagged graph
  - Data files: *_instr_v6_graphs.pkl
  - Checkpoint default: model_instr_v6.pt

Usage:
    python train_instr_v6.py --epochs 30 --hidden 64
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
        # input: opcode_embed(128) + const_mag(1) + taint(1) = 130
        self.conv1 = RGCNConv(embed_dim + 2, hidden, num_relations=3)
        self.conv2 = RGCNConv(hidden,        hidden, num_relations=3)
        gate_nn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )
        self.pool = AttentionalAggregation(gate_nn=gate_nn)
        self.lin  = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_type, batch):
        opcode_ids = x[:, 0].long()
        const_mag  = x[:, 1].unsqueeze(-1)
        taint_val  = x[:, 2].unsqueeze(-1)   # 0.0=clean, 1.0=source, 0.5/0.25/0.125=propagated
        h = torch.cat([self.embed(opcode_ids), const_mag, taint_val], dim=-1)
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
        edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
        edge_type  = torch.tensor(g["edge_type"],  dtype=torch.long)
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
    ap.add_argument("--checkpoint", type=str,   default="model_instr_v6.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    for split in ["train", "valid", "test"]:
        if not (DATA / f"{split}_instr_v6_graphs.pkl").exists():
            print(f"Missing data/{split}_instr_v6_graphs.pkl -- run preprocess_instr_v6.py first.")
            sys.exit(1)

    print("Loading graphs ...")
    train_data = load_graphs(DATA / "train_instr_v6_graphs.pkl")
    valid_data = load_graphs(DATA / "valid_instr_v6_graphs.pkl")
    test_data  = load_graphs(DATA / "test_instr_v6_graphs.pkl")
    print(f"  train={len(train_data)}  valid={len(valid_data)}  test={len(test_data)}")

    # Coverage diagnostics — source nodes (taint >= 1.0) and propagated nodes (>0)
    source_graphs  = [d for d in train_data if (d.x[:, 2] >= 1.0).any()]
    tainted_graphs = [d for d in train_data if (d.x[:, 2] > 0.0).any()]
    n = len(train_data)
    print(f"  train graphs with ≥1 source node (flag=1.0): "
          f"{len(source_graphs)} / {n} ({len(source_graphs)/n*100:.1f}%)")
    print(f"  train graphs with any taint (incl. propagated): "
          f"{len(tainted_graphs)} / {n} ({len(tainted_graphs)/n*100:.1f}%)")
    if tainted_graphs:
        mean_taint = sum(d.x[:, 2].mean().item() for d in tainted_graphs) / len(tainted_graphs)
        print(f"  mean taint value across tainted graphs: {mean_taint:.4f}")

    # Flag correlation with ground-truth labels — tells us if the pattern is discriminative
    flagged = [(d.x[:, 2] > 0).any().item() for d in train_data]
    labels  = [int(d.y.item()) for d in train_data]
    fv = sum(1 for f, l in zip(flagged, labels) if f and l)       # flagged & vuln
    fc = sum(1 for f, l in zip(flagged, labels) if f and not l)   # flagged & clean
    uv = sum(1 for f, l in zip(flagged, labels) if not f and l)   # unflagged & vuln
    n_f, n_v = fv + fc, fv + uv
    if n_f:
        print(f"  flag precision P(vuln|flagged):    {fv}/{n_f} ({fv/n_f*100:.1f}%)")
    if n_v:
        print(f"  flag recall    P(flagged|vuln):    {fv}/{n_v} ({fv/n_v*100:.1f}%)")
        print(f"  ceiling miss rate (unflagged vuln): {uv}/{n_v} ({uv/n_v*100:.1f}%)"
              "  <-- functions the model cannot see even in principle")

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
          f"in_dim={args.embed_dim+2}, hidden={args.hidden}, relations=3)  "
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

    print("--- Comparison ---")
    print(f"  §7  Instruction-level GNN baseline:                    56.53%")
    print(f"  §13 + Perfograph + call categorization:                58.75% (best single run)")
    print(f"  §14 + VSDG memory ordering edges:                      57.47%")
    print(f"  §15 + register name embedding:                         57.47%")
    print(f"  §16 + static analysis flags (binary):                  57.15%")
    print(f"  §17 + taint propagation + extended patterns (this):    {test_acc:.2%}")

    delta_vs13 = test_acc - 0.5875
    delta_vs16 = test_acc - 0.5715
    if delta_vs13 > 0.005:
        print(f"\n  +{delta_vs13:.2%} over §13 — taint propagation adds genuine signal")
    elif delta_vs16 > 0.005:
        print(f"\n  +{delta_vs16:.2%} over §16 — propagation helps within-graph density")
    elif delta_vs13 > -0.005:
        print(f"\n  Within noise of §13 — coverage/density still the bottleneck")
    else:
        print(f"\n  Below §13 — check taint coverage stats; source patterns still too sparse")


if __name__ == "__main__":
    main()
