#!/usr/bin/env python3
"""
train_slice_pdg_v6.py — Train the PDG slice GNN on Joern-preprocessed PrimeVul (§26).

§26 experiment: same SlicePDGGNN architecture as §12/§25, but trained on Joern
CPG graphs instead of LLVM IR graphs.  Joern's fuzzy parser achieves ~95% PrimeVul
coverage vs ~34% for the clang pipeline, and more importantly eliminates the 2x
survival bias (vuln 21.8% vs benign 37%) that made §25 systematically undersample
the complex stateful functions where real vulnerabilities live.

Node vocabulary: 16 Joern CPG tokens (vs 110 LLVM opcodes for §12/§25).
  NOT compatible with LLVM IR model checkpoints — different feature space.

Prerequisites:
    python preprocess_primevul_joern.py --max-benign 21000

Usage:
    python train_slice_pdg_v6.py                          # 30ep h=64
    python train_slice_pdg_v6.py --epochs 60 --hidden 128
"""

import argparse
import pickle
import statistics
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

VOCAB_SIZE = 16   # Joern CPG token vocabulary (preprocess_joern.VOCAB_SIZE)


# ---------------------------------------------------------------------------
# Model — same architecture as §12, vocab adapted for Joern token space
# ---------------------------------------------------------------------------

class SlicePDGGNN(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, embed_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim)
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
        x          = torch.tensor(g["x"],         dtype=torch.long)
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
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    ap.add_argument("--epochs",     type=int,   default=30)
    ap.add_argument("--hidden",     type=int,   default=64)
    ap.add_argument("--embed-dim",  type=int,   default=128)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int,   default=32)
    ap.add_argument("--checkpoint", type=str,   default="model_slice_pdg_v6.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    for split in ["train", "valid", "test"]:
        p = DATA / f"{split}_joern_graphs.pkl"
        if not p.exists():
            print(f"Missing {p}")
            print("Run: python preprocess_primevul_joern.py --max-benign 21000")
            sys.exit(1)

    print("Loading Joern PrimeVul PDG slice graphs ...")
    train_data = load_graphs(DATA / "train_joern_graphs.pkl")
    valid_data = load_graphs(DATA / "valid_joern_graphs.pkl")
    test_data  = load_graphs(DATA / "test_joern_graphs.pkl")
    print(f"  train={len(train_data)}  valid={len(valid_data)}  test={len(test_data)}")

    node_counts = [d.x.shape[0] for d in train_data]
    print(f"  Train slice sizes: mean={statistics.mean(node_counts):.0f}  "
          f"median={statistics.median(node_counts):.0f}  max={max(node_counts)}")

    vuln_train  = sum(1 for d in train_data if d.y.item() == 1)
    fixed_train = len(train_data) - vuln_train
    ratio       = fixed_train / vuln_train if vuln_train > 0 else 1.0
    pos_weight  = torch.tensor([ratio]).to(device)
    print(f"  train class balance: {vuln_train} vuln / {fixed_train} benign  "
          f"(ratio 1:{ratio:.1f})")
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
    print(f"PrimeVul test accuracy: {test_acc:.2%}")
    print(f"Checkpoint saved to: {checkpoint.resolve()}\n")

    print("--- §26 Results ---")
    print(f"  §25 PDG slice, PrimeVul via clang (34% coverage): TBD PrimeVul test")
    print(f"  §26 PDG slice, PrimeVul via Joern (~95% coverage): {test_acc:.2%} PrimeVul test")
    print(f"\n  Next: run eval_all_models.py (requires Joern-preprocessed scarnet)")
    print(f"  Devign cross-eval: preprocess Devign with Joern, "
          f"then rerun with --checkpoint {checkpoint}")


if __name__ == "__main__":
    main()
