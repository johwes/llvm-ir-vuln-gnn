#!/usr/bin/env python3
"""
train_slice_pdg_v5.py — Train the PDG slice GNN on PrimeVul (§25).

§25 experiment: PrimeVul (ICSE 2025, arXiv:2403.18624) applies LLM-assisted
relabeling to real CVE commits, achieving function-level label accuracy ~3x
better than Devign's commit-level approach.  Training on PrimeVul targets the
55-58% Devign accuracy ceiling caused by ~10-20% Devign label noise.

Same SlicePDGGNN architecture as §12 (best Devign/scarnet model).
Trains on PrimeVul, validates on PrimeVul, reports both PrimeVul test accuracy
and Devign test accuracy (cross-dataset comparison against §12's 56.48%).

Prerequisites:
    pip install datasets
    python preprocess_primevul.py --max-benign 21000   # balanced dataset
    python preprocess_slice_pdg.py                     # for Devign cross-eval

Usage:
    python train_slice_pdg_v5.py                          # default 30ep h=64
    python train_slice_pdg_v5.py --epochs 60 --hidden 128 # extended
    python train_slice_pdg_v5.py --no-devign-eval          # skip Devign cross-eval

Class imbalance: PrimeVul is ~1:33 (vuln:benign).  With --max-benign 21000
it becomes ~1:3 (pos_weight ≈ 3).  pos_weight is computed from the training
split automatically.
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

VOCAB_SIZE = 110


# ---------------------------------------------------------------------------
# Model — identical architecture to §12 (SlicePDGGNN in train_slice_pdg.py)
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
    ap.add_argument("--epochs",         type=int,   default=30)
    ap.add_argument("--hidden",         type=int,   default=64)
    ap.add_argument("--embed-dim",      type=int,   default=128)
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--batch-size",     type=int,   default=32)
    ap.add_argument("--checkpoint",     type=str,   default="model_slice_pdg_v5.pt")
    ap.add_argument("--no-devign-eval", action="store_true",
                    help="Skip Devign cross-dataset evaluation at the end")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # -- Verify PrimeVul data exists ------------------------------------------
    for split in ["train", "valid", "test"]:
        p = DATA / f"{split}_primevul_graphs.pkl"
        if not p.exists():
            print(f"Missing {p}")
            print("Run: python preprocess_primevul.py --max-benign 21000")
            sys.exit(1)

    # -- Load PrimeVul splits --------------------------------------------------
    print("Loading PrimeVul PDG slice graphs ...")
    train_data = load_graphs(DATA / "train_primevul_graphs.pkl")
    valid_data = load_graphs(DATA / "valid_primevul_graphs.pkl")
    test_data  = load_graphs(DATA / "test_primevul_graphs.pkl")
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

    # -- PrimeVul test accuracy -----------------------------------------------
    print(f"\nLoading best checkpoint ({checkpoint}) ...")
    model.load_state_dict(torch.load(checkpoint, map_location=device,
                                     weights_only=True))
    primevul_test_acc = evaluate(model, test_loader, device)
    print(f"PrimeVul test accuracy: {primevul_test_acc:.2%}")

    # -- Devign cross-dataset evaluation --------------------------------------
    devign_test_acc = None
    if not args.no_devign_eval:
        devign_pkl = DATA / "test_slice_pdg_graphs.pkl"
        if devign_pkl.exists():
            print(f"\nCross-dataset: evaluating on Devign test set ...")
            devign_data   = load_graphs(devign_pkl)
            devign_loader = DataLoader(devign_data, batch_size=args.batch_size)
            devign_vuln   = sum(1 for d in devign_data if d.y.item() == 1)
            devign_benign = len(devign_data) - devign_vuln
            print(f"  Devign test: {len(devign_data)} functions  "
                  f"({devign_vuln} vuln / {devign_benign} benign)")
            devign_test_acc = evaluate(model, devign_loader, device)
            print(f"  Devign test accuracy: {devign_test_acc:.2%}")
        else:
            print(f"\nSkipping Devign cross-eval (missing {devign_pkl})")
            print("  Run: python preprocess_slice_pdg.py")

    print(f"\nCheckpoint saved to: {checkpoint.resolve()}\n")

    # -- Results summary -------------------------------------------------------
    print("--- Results ---")
    print(f"  §12 PDG slice (Devign train+test):          56.48% Devign")
    print(f"  §25 PDG slice (PrimeVul train):             {primevul_test_acc:.2%} PrimeVul")
    if devign_test_acc is not None:
        print(f"  §25 cross-dataset:                          {devign_test_acc:.2%} Devign")
        delta = devign_test_acc - 0.5648
        sign  = "+" if delta >= 0 else ""
        print(f"  vs §12 baseline:                            {sign}{delta:.2%}")

        if devign_test_acc >= 0.63:
            print("\n  Breaks through the Devign noise ceiling — cleaner labels generalise")
        elif devign_test_acc >= 0.58:
            print("\n  Improvement over §12 — PrimeVul labels provide better training signal")
        elif devign_test_acc >= 0.5648:
            print("\n  Marginal improvement — PrimeVul distribution differs from Devign")
        else:
            print("\n  Below §12 — distribution shift dominates; consider fine-tuning on Devign")


if __name__ == "__main__":
    main()
