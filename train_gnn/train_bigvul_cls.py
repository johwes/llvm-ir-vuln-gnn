#!/usr/bin/env python3
"""
train_bigvul_cls.py — §21 standard binary classifier on BigVul (instruction-level v2).

Architecture: InstructionGNN from train_instr_v2.py (identical weight format)
  Embedding(111,128) + Perfograph const_magnitude + RGCNConv(3 rels)×2
  + AttentionalAggregation + Linear(hidden→1)

This experiment tests whether BigVul's CVE-sourced, function-level labels
produce a better-calibrated model than Devign's commit-level labels.

Data sources:
  BigVul only (default):
    data/bigvul_cls_{train,valid,test}_instr_v2_graphs.pkl
  Combined with Devign (--combine-devign):
    also loads data/{train,valid}_instr_v2_graphs.pkl

Usage:
    # BigVul only (saves model_bigvul_cls.pt)
    python train_bigvul_cls.py --epochs 30 --hidden 64

    # BigVul + Devign combined (saves model_bigvul_combined.pt)
    python train_bigvul_cls.py --epochs 30 --hidden 64 --combine-devign
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

HERE = Path(__file__).parent
DATA = HERE / "data"

sys.path.insert(0, str(HERE))
from train_instr_v2 import InstructionGNN, VOCAB_SIZE


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
    """Return (accuracy, balanced_accuracy).

    Balanced accuracy = mean(TPR, TNR) — robust to class imbalance.
    Raw accuracy is kept for display; balanced accuracy drives checkpoint saving.
    """
    model.eval()
    tp = fp = tn = fn = 0
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        preds  = (logits > 0).long()
        labels = batch.y.squeeze().long()
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
    total = tp + fp + tn + fn
    acc      = (tp + tn) / total if total > 0 else 0.0
    tpr      = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall / sensitivity
    tnr      = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity
    bal_acc  = (tpr + tnr) / 2
    return acc, bal_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",         type=int,   default=30)
    ap.add_argument("--hidden",         type=int,   default=64)
    ap.add_argument("--embed-dim",      type=int,   default=128)
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--batch-size",     type=int,   default=32)
    ap.add_argument("--combine-devign", action="store_true",
                    help="Concatenate Devign train/valid splits with BigVul")
    ap.add_argument("--checkpoint",     type=str,   default=None,
                    help="Checkpoint filename (default: auto-named by data source)")
    args = ap.parse_args()

    if args.checkpoint is None:
        args.checkpoint = ("model_bigvul_combined.pt" if args.combine_devign
                           else "model_bigvul_cls.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # -- Load BigVul splits -------------------------------------------------------
    for split in ["train", "valid", "test"]:
        p = DATA / f"bigvul_cls_{split}_instr_v2_graphs.pkl"
        if not p.exists():
            print(f"Missing {p}\nRun: python preprocess_bigvul_cls.py")
            sys.exit(1)

    print("Loading BigVul graphs ...")
    train_data = load_graphs(DATA / "bigvul_cls_train_instr_v2_graphs.pkl")
    valid_data = load_graphs(DATA / "bigvul_cls_valid_instr_v2_graphs.pkl")
    test_data  = load_graphs(DATA / "bigvul_cls_test_instr_v2_graphs.pkl")
    print(f"  BigVul  train={len(train_data):,}  valid={len(valid_data):,}"
          f"  test={len(test_data):,}")

    # -- Optionally combine with Devign ------------------------------------------
    if args.combine_devign:
        for split in ["train", "valid"]:
            p = DATA / f"{split}_instr_v2_graphs.pkl"
            if not p.exists():
                print(f"Missing {p}\nRun: python preprocess_instr_v2.py")
                sys.exit(1)
        print("Loading Devign graphs ...")
        dv_train = load_graphs(DATA / "train_instr_v2_graphs.pkl")
        dv_valid = load_graphs(DATA / "valid_instr_v2_graphs.pkl")
        train_data = train_data + dv_train
        valid_data = valid_data + dv_valid
        print(f"  Combined  train={len(train_data):,}  valid={len(valid_data):,}")
        print("  (test set remains BigVul-only for clean evaluation)")

    # -- Class balance -----------------------------------------------------------
    vuln_train  = sum(1 for d in train_data if d.y.item() == 1)
    fixed_train = len(train_data) - vuln_train
    pos_weight  = torch.tensor([fixed_train / max(vuln_train, 1)]).to(device)
    print(f"\n  train class balance: {vuln_train:,} vuln / {fixed_train:,} clean"
          f"  (pos_weight={pos_weight.item():.3f})")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size)

    model     = InstructionGNN(VOCAB_SIZE, args.embed_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: InstructionGNN(vocab={VOCAB_SIZE}, embed={args.embed_dim},"
          f" hidden={args.hidden}, relations=3)  params={n_params:,}\n")

    checkpoint = Path(args.checkpoint)
    best_val   = 0.0

    print(f"{'Epoch':>5}  {'Loss':>8}  {'Val Acc':>8}  {'Bal Acc':>8}  {'':>8}")
    print("-" * 48)

    for epoch in range(1, args.epochs + 1):
        loss              = train_epoch(model, train_loader, optimizer, device, pos_weight)
        val_acc, bal_acc  = evaluate(model, valid_loader, device)
        scheduler.step()

        marker = ""
        if bal_acc > best_val:
            best_val = bal_acc
            torch.save(model.state_dict(), checkpoint)
            marker = "<- best"
        print(f"{epoch:>5}  {loss:>8.4f}  {val_acc:>8.2%}  {bal_acc:>8.2%}  {marker}")

    print(f"\nLoading best checkpoint ({checkpoint}) ...")
    model.load_state_dict(torch.load(checkpoint, map_location=device,
                                     weights_only=True))
    test_acc, test_bal = evaluate(model, test_loader, device)
    label = "BigVul+Devign" if args.combine_devign else "BigVul"
    print(f"Test accuracy ({label} test split): {test_acc:.2%}  (balanced: {test_bal:.2%})")
    print(f"Checkpoint: {checkpoint.resolve()}\n")

    neg_count   = sum(1 for d in test_data if d.y.item() == 0)
    majority_bl = neg_count / len(test_data) if test_data else 0.0
    print(f"  Majority-class baseline (predict all negative): {majority_bl:.2%}")
    print(f"  This run ({label}):  raw={test_acc:.2%}  balanced={test_bal:.2%}")
    print()
    print("NOTE: accuracy on this imbalanced dataset is not directly comparable to")
    print("      Devign-trained models (CodeBERT 63.43%, §13 58.75% were on ~50/50 split).")
    print("      Use eval_all_models.py --scarnet for a fair real-world comparison.")
    print()
    print("Next: run eval_all_models.py --scarnet to compare against Devign-trained models")
    print()


if __name__ == "__main__":
    main()
