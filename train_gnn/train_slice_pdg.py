#!/usr/bin/env python3
"""
train_slice_pdg.py — Train a GNN classifier on PDG backward-slice graphs.

§12 experiment: §11 (DFG-only slice, 30ep) produced 56.64% — below the §7
baseline of 58.00%. Root cause: DFG-only slicing misses guard conditions.
`if (n < sizeof(buf)) memcpy(...)` — the icmp+br nodes guarding the call
have no DFG edge into the sink. Both guarded and unguarded versions produce
identical DFG slices.

This experiment uses PDG slices (DFG + control dependence). The control
dependence of a node v = the terminator (br/switch) instructions of all
CFG predecessor blocks of v's basic block. In LLVM IR, br already has a
DFG edge from its condition (icmp), so adding br automatically pulls the
icmp guard and its operands into the slice via the next DFG BFS iteration.

Expected: PDG slices distinguish guarded from unguarded dangerous calls.
Baseline: §11 56.64%, §7 58.00%, CodeBERT 63.43%.

Architecture: identical to train_slice.py (RGCN + AttentionalAggregation).
PDG graphs have the same format — same opcode vocab, same 3 edge types.

Usage:
    python train_slice_pdg.py --epochs 30 --hidden 64    # baseline
    python train_slice_pdg.py --epochs 60 --hidden 128   # extended
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
# Model — identical to train_slice.py
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
    ap.add_argument("--checkpoint", type=str,   default="model_slice_pdg.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

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
    print(f"  §7  Instruction-level GNN (full graph, 30ep h=64):        58.00%")
    print(f"  §11 Slice-GNN (backward DFG from sinks, 30ep h=64):       56.64%")
    print(f"  §12 Slice-PDG-GNN (DFG + control deps, this run):         {test_acc:.2%}")

    if test_acc >= 0.62:
        print("\n  PDG slice reaches VulPathFinder territory (61%) -- "
              "control dependence is the missing structural signal")
    elif test_acc >= 0.58:
        print("\n  Improvement over §7 baseline -- PDG slicing adds discriminative "
              "signal beyond the full-graph classifier; consider longer training")
    elif test_acc >= 0.5664:
        print("\n  Improvement over §11 DFG-only slice -- control deps help but "
              "do not fully close the gap; review slice size stats")
    else:
        print("\n  No improvement over §11 -- check preprocess_slice_pdg.py output "
              "for slice size distribution and sliced/fallback fraction")


if __name__ == "__main__":
    main()
