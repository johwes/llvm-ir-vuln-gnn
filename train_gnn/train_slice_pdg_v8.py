#!/usr/bin/env python3
"""
train_slice_pdg_v8.py — §28 Juliet pretrain → Devign RankNet fine-tune.

§27 established that Juliet pretraining transfers to Devign without
catastrophic forgetting, but Phase 2 BCE on Devign's noisy labels
does not move the scarnet ceiling — 11/13 parity with §12.

§28 replaces Phase 2's binary cross-entropy with a pairwise RankNet loss:
  For each mini-batch, form all (vuln, benign) pairs and maximise
    P(score(vuln) > score(benign))  =  σ(score_v - score_b)
  This is directly the ranking objective — the model is trained to rank
  suspicious functions above clean ones, which is exactly what scarnet
  measures, rather than to cross a 0/1 threshold under noisy labels.

  Advantages over BCE:
  - Label noise hurts BCE when the model is confident and wrong.
    RankNet loss depends on the *ordering*, which survives label noise
    better — if a "vuln" label is wrong, the model just wastes a pair,
    it does not get a strong wrong-direction gradient.
  - Directly aligns training objective with scarnet evaluation.

Architecture: SlicePDGGNN_v7 — same RGCN backbone, multi-feature x(N,3).
The model is identical to §27; only the Phase 2 loss changes.

Two-phase training:
  Phase 1 — Juliet BCE (clean signal, same as §27)
             Output: model_juliet_pretrain.pt  (shared with §27 if already run)
  Phase 2 — Devign RankNet (ranking objective)
             Output: model_slice_pdg_v8.pt

Usage:
    python train_slice_pdg_v8.py --help
    python train_slice_pdg_v8.py --pretrain-epochs 20 --finetune-epochs 40
    python train_slice_pdg_v8.py --finetune-only   # skip Phase 1 if pretrain ckpt exists
    python train_slice_pdg_v8.py --pretrain-only   # Phase 1 only
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

# Reuse SlicePDGGNN_v7 — identical architecture
from train_slice_pdg_v7 import SlicePDGGNN_v7, load_graphs

HERE  = Path(__file__).parent
DATA  = HERE / "data"

VOCAB_SIZE = 110
N_SCALAR   = 2


# ---------------------------------------------------------------------------
# RankNet pairwise loss
# ---------------------------------------------------------------------------

def ranknet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Pairwise RankNet loss for a mini-batch.

    For each (i, j) pair where label[i]=1 (vuln) and label[j]=0 (benign):
      loss += -log σ(score[i] - score[j])

    Returns the mean over all pairs in the batch.  If the batch is all-one-class
    (no pairs possible), returns a zero-gradient BCE loss on the batch so training
    still makes progress on lopsided batches.

    Args:
        scores: (B,) raw logits from the model (not probabilities)
        labels: (B,) binary labels, float
    """
    vuln_mask  = labels > 0.5
    benign_mask = ~vuln_mask

    vuln_scores   = scores[vuln_mask]
    benign_scores = scores[benign_mask]

    if vuln_scores.numel() == 0 or benign_scores.numel() == 0:
        # Fallback to BCE for single-class batches — ensures gradient flow
        return F.binary_cross_entropy_with_logits(scores, labels)

    # Broadcast: (n_vuln, 1) - (1, n_benign) → (n_vuln, n_benign) pair matrix
    diff = vuln_scores.unsqueeze(1) - benign_scores.unsqueeze(0)
    loss = F.softplus(-diff)   # -log σ(diff)  =  log(1 + exp(-diff))
    return loss.mean()


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch_bce(model, loader, optimizer, device, pos_weight=None):
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


def train_epoch_ranknet(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        labels = batch.y.squeeze()
        loss   = ranknet_loss(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / n_batches if n_batches else 0.0


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


def run_phase_bce(label, model, train_data, valid_data,
                  epochs, lr, batch_size, device, checkpoint,
                  lr_step=10, lr_gamma=0.5):
    """BCE training — used for Phase 1 (Juliet)."""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    n_vuln  = sum(1 for d in train_data if d.y.item() == 1)
    n_fixed = len(train_data) - n_vuln
    ratio   = n_fixed / n_vuln if n_vuln > 0 else 1.0
    pw      = torch.tensor([ratio]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    _phase_header(label, train_data, valid_data, n_vuln, n_fixed, ratio)

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        loss    = train_epoch_bce(model, train_loader, optimizer, device, pw)
        val_acc = evaluate(model, valid_loader, device)
        scheduler.step()
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), checkpoint)
            marker = "<- best"
        print(f"{epoch:>5}  {loss:>8.4f}  {val_acc:>8.2%}  {marker}")

    print(f"\n  Best val acc: {best_val:.2%}  (checkpoint: {checkpoint})")
    return best_val


def run_phase_ranknet(label, model, train_data, valid_data,
                      epochs, lr, batch_size, device, checkpoint,
                      lr_step=10, lr_gamma=0.5):
    """RankNet training — used for Phase 2 (Devign)."""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    n_vuln  = sum(1 for d in train_data if d.y.item() == 1)
    n_fixed = len(train_data) - n_vuln
    ratio   = n_fixed / n_vuln if n_vuln > 0 else 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    _phase_header(label, train_data, valid_data, n_vuln, n_fixed, ratio)
    print("  Loss: pairwise RankNet (not BCE — pairs per batch, not per sample)")

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        loss    = train_epoch_ranknet(model, train_loader, optimizer, device)
        val_acc = evaluate(model, valid_loader, device)
        scheduler.step()
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), checkpoint)
            marker = "<- best"
        print(f"{epoch:>5}  {loss:>8.4f}  {val_acc:>8.2%}  {marker}")

    print(f"\n  Best val acc: {best_val:.2%}  (checkpoint: {checkpoint})")
    return best_val


def _phase_header(label, train_data, valid_data, n_vuln, n_fixed, ratio):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  train={len(train_data)}  valid={len(valid_data)}")
    print(f"  class balance: {n_vuln} vuln / {n_fixed} benign  (ratio={ratio:.2f})")
    print(f"{'─'*55}")
    print(f"{'Epoch':>5}  {'Loss':>8}  {'Val Acc':>8}")
    print("-" * 28)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    ap.add_argument("--pretrain-epochs", type=int,   default=20,
                    help="Epochs for Phase 1 Juliet BCE pretraining")
    ap.add_argument("--finetune-epochs", type=int,   default=40,
                    help="Epochs for Phase 2 Devign RankNet fine-tune")
    ap.add_argument("--hidden",          type=int,   default=64)
    ap.add_argument("--embed-dim",       type=int,   default=128)
    ap.add_argument("--lr",              type=float, default=1e-3,
                    help="Learning rate for Phase 1")
    ap.add_argument("--finetune-lr",     type=float, default=1e-4,
                    help="Lower LR for Phase 2 (RankNet on Devign)")
    ap.add_argument("--batch-size",      type=int,   default=32,
                    help="Batch size (larger = more pairs per RankNet step)")
    ap.add_argument("--pretrain-only",   action="store_true")
    ap.add_argument("--finetune-only",   action="store_true")
    ap.add_argument("--pretrain-ckpt",   type=str,
                    default="model_juliet_pretrain.pt",
                    help="Phase 1 checkpoint (shared with §27 if already trained)")
    ap.add_argument("--checkpoint",      type=str,
                    default="model_slice_pdg_v8.pt",
                    help="Final §28 checkpoint")
    ap.add_argument("--devign-train",    type=str, default=None)
    ap.add_argument("--devign-valid",    type=str, default=None)
    ap.add_argument("--devign-test",     type=str, default=None)
    args = ap.parse_args()

    if args.pretrain_only and args.finetune_only:
        print("ERROR: --pretrain-only and --finetune-only are mutually exclusive.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n§28 Juliet BCE pretrain → Devign RankNet fine-tune")
    print(f"Device: {device}")

    juliet_train = DATA / "train_juliet_graphs.pkl"
    juliet_valid = DATA / "valid_juliet_graphs.pkl"

    def _devign_path(split, override):
        if override:
            return Path(override)
        for candidate in [
            DATA / f"{split}_slice_pdg_v7_graphs.pkl",
            DATA / f"{split}_slice_pdg_graphs.pkl",
        ]:
            if candidate.exists():
                return candidate
        return DATA / f"{split}_slice_pdg_graphs.pkl"

    devign_train_path = _devign_path("train", args.devign_train)
    devign_valid_path = _devign_path("valid", args.devign_valid)
    devign_test_path  = _devign_path("test",  args.devign_test)

    pretrain_ckpt = Path(args.pretrain_ckpt)
    final_ckpt    = Path(args.checkpoint)

    model = SlicePDGGNN_v7(VOCAB_SIZE, args.embed_dim, args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SlicePDGGNN_v7(vocab={VOCAB_SIZE}, embed={args.embed_dim}, "
          f"hidden={args.hidden}, n_scalar={N_SCALAR})  params={n_params:,}")

    # =========================================================================
    # Phase 1 — Juliet BCE pretraining
    # =========================================================================
    if not args.finetune_only:
        for p in [juliet_train, juliet_valid]:
            if not p.exists():
                print(f"\nMissing: {p}")
                print("Run: python preprocess_juliet.py --workers 8")
                sys.exit(1)

        print("\n-- Loading Juliet graphs --")
        j_train = load_graphs(juliet_train)
        j_valid = load_graphs(juliet_valid)
        nc = [d.x.shape[0] for d in j_train]
        print(f"  train={len(j_train)}  valid={len(j_valid)}")
        print(f"  Slice sizes: mean={statistics.mean(nc):.0f}  "
              f"median={statistics.median(nc):.0f}  max={max(nc)}")

        # If the §27 pretrain checkpoint already exists and we haven't been
        # asked to re-run Phase 1, reuse it to save time.
        if pretrain_ckpt.exists() and not args.pretrain_only:
            print(f"\n  Phase 1 checkpoint already exists: {pretrain_ckpt}")
            print(f"  Loading and skipping Phase 1 training.  "
                  f"Pass --pretrain-only to force re-training.")
            model.load_state_dict(torch.load(pretrain_ckpt, map_location=device,
                                              weights_only=True))
        else:
            run_phase_bce(
                label="Phase 1 — Juliet BCE pretraining",
                model=model,
                train_data=j_train,
                valid_data=j_valid,
                epochs=args.pretrain_epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                device=device,
                checkpoint=pretrain_ckpt,
            )

        if args.pretrain_only:
            print(f"\nPhase 1 complete.  Checkpoint: {pretrain_ckpt.resolve()}")
            print("Re-run with --finetune-only to continue to Phase 2.")
            return

    # =========================================================================
    # Phase 2 — Devign RankNet fine-tune
    # =========================================================================
    if not args.pretrain_only:
        for p in [devign_train_path, devign_valid_path, devign_test_path]:
            if not p.exists():
                print(f"\nMissing: {p}")
                print("Run: python preprocess_slice_pdg.py")
                sys.exit(1)

        if args.finetune_only:
            if pretrain_ckpt.exists():
                print(f"\nLoading pretrained weights from {pretrain_ckpt} ...")
                model.load_state_dict(torch.load(pretrain_ckpt, map_location=device,
                                                  weights_only=True))
            else:
                print(f"WARNING: --finetune-only but {pretrain_ckpt} not found. "
                      f"Starting from random init.")

        print(f"\n-- Loading Devign graphs --")
        print(f"  train: {devign_train_path.name}")
        print(f"  valid: {devign_valid_path.name}")
        print(f"  test:  {devign_test_path.name}")
        d_train = load_graphs(devign_train_path)
        d_valid = load_graphs(devign_valid_path)
        d_test  = load_graphs(devign_test_path)
        feat_cols = d_train[0].x.shape[1] if d_train else 1
        print(f"  train={len(d_train)}  valid={len(d_valid)}  test={len(d_test)}")
        print(f"  Feature cols: {feat_cols}"
              f"  {'(§12 compat, cols 1&2 zero-padded)' if feat_cols == 1 else '(multi-feature)'}")

        print(f"\n  Note: Phase 2 uses pairwise RankNet loss.")
        print(f"  Batch size {args.batch_size} → up to {args.batch_size//2 * args.batch_size//2} pairs/step.")

        run_phase_ranknet(
            label="Phase 2 — Devign RankNet fine-tune",
            model=model,
            train_data=d_train,
            valid_data=d_valid,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            batch_size=args.batch_size,
            device=device,
            checkpoint=final_ckpt,
        )

        print(f"\n-- Devign test evaluation --")
        model.load_state_dict(torch.load(final_ckpt, map_location=device,
                                          weights_only=True))
        test_loader = DataLoader(d_test, batch_size=args.batch_size)
        test_acc    = evaluate(model, test_loader, device)

        print(f"\n{'='*55}")
        print(f"  §28 results summary")
        print(f"{'='*55}")
        print(f"  Phase 1: Juliet BCE  (structural prior)")
        print(f"  Phase 2: Devign RankNet  ({devign_train_path.name})")
        print(f"  Devign test acc: {test_acc:.2%}")
        print(f"")
        print(f"  Baselines:")
        print(f"    §12  PDG BCE (Devign only):           56.48%")
        print(f"    §27  Juliet + Devign BCE:             56.12%")
        print(f"    §28  Juliet + Devign RankNet:         {test_acc:.2%}  <-- this run")
        print(f"")
        print(f"  NOTE: Devign accuracy is not the primary metric.")
        print(f"  RankNet trains an ordering, not a classifier — accuracy may")
        print(f"  sit near 50% while ranking quality improves on scarnet.")
        print(f"  Run eval_all_models.py --scarnet to measure real quality.")
        print(f"{'='*55}")
        print(f"  Checkpoint: {final_ckpt.resolve()}")


if __name__ == "__main__":
    main()
