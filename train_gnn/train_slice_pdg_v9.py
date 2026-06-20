#!/usr/bin/env python3
"""
train_slice_pdg_v9.py — §30 Juliet positives + clean real-C negatives.

§29 showed the Juliet-only model saturates on real production code (91–100%
on everything) because its negatives were synthetic Juliet good functions —
nothing like real server C.  The Devign fine-tune provides calibration but
re-introduces commit-history fingerprinting and label noise.

§30 breaks this by replacing Devign entirely:

  Positives:  Juliet bad functions  (unguarded sinks, zero label noise)
  Negatives:  Juliet good functions + zlib + musl + SQLite functions (label 0)

The model learns:
  "this is what an unguarded dangerous sink looks like" (from Juliet bad)
  "this is what clean real C looks like"               (from real negatives)

No Devign.  No commit history.  No label noise.

Two-phase training:
  Phase 1 — Juliet bad vs Juliet good  (clean synthetic structural prior)
             Output: model_juliet_pretrain.pt  (reused from §27/§28 if present)
  Phase 2 — Juliet bad vs (Juliet good + real clean C)
             Same model, continued training on the augmented negative set.
             Output: model_slice_pdg_v9.pt

Architecture: SlicePDGGNN_v7 — identical to §27/§28.

Prerequisites:
    # Juliet graphs (reuse from §27/§28 if already built):
    python preprocess_juliet.py --workers 8

    # Clean C negatives (new):
    python preprocess_clean_negatives.py --workers 4

Usage:
    python train_slice_pdg_v9.py --help
    python train_slice_pdg_v9.py --finetune-only   # skip Phase 1 if ckpt exists
    python train_slice_pdg_v9.py --finetune-epochs 40
"""

import argparse
import pickle
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from train_slice_pdg_v7 import SlicePDGGNN_v7, load_graphs, run_phase

HERE = Path(__file__).parent
DATA = HERE / "data"

VOCAB_SIZE = 110
N_SCALAR   = 2


# ---------------------------------------------------------------------------
# Data loading — merge Juliet good + real clean C negatives
# ---------------------------------------------------------------------------

def load_combined(juliet_pkl: Path,
                  clean_neg_pkl: Path,
                  rng: random.Random,
                  max_nodes: int = 500) -> list[Data]:
    """
    Load Juliet graphs and clean-negative graphs, combine, shuffle.
    Juliet pkl contains both vuln (y=1) and benign (y=0).
    Clean-neg pkl is all y=0.
    Graphs with more than max_nodes nodes are dropped — outliers (e.g. the
    Lua interpreter main loop at 6,123 nodes) would dominate VRAM per batch.
    Returns a combined, shuffled list.
    """
    juliet = load_graphs(juliet_pkl)
    clean  = load_graphs(clean_neg_pkl)
    combined = juliet + clean

    before = len(combined)
    combined = [g for g in combined if g.x.shape[0] <= max_nodes]
    dropped = before - len(combined)
    if dropped:
        print(f"  Dropped {dropped} graphs exceeding {max_nodes} nodes")

    rng.shuffle(combined)
    return combined


def load_combined_splits(rng: random.Random,
                         juliet_train: Path,
                         juliet_valid: Path,
                         clean_train: Path,
                         clean_valid: Path,
                         max_nodes: int = 500) -> tuple[list, list]:
    train = load_combined(juliet_train, clean_train, rng, max_nodes)
    valid = load_combined(juliet_valid, clean_valid, rng, max_nodes)
    return train, valid


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
                    help="Phase 1 Juliet-only BCE epochs (skipped if ckpt exists)")
    ap.add_argument("--finetune-epochs", type=int,   default=40,
                    help="Phase 2 Juliet+real-clean BCE epochs")
    ap.add_argument("--hidden",          type=int,   default=64)
    ap.add_argument("--embed-dim",       type=int,   default=128)
    ap.add_argument("--lr",              type=float, default=1e-3,
                    help="Phase 1 learning rate")
    ap.add_argument("--finetune-lr",     type=float, default=3e-4,
                    help="Phase 2 learning rate")
    ap.add_argument("--batch-size",      type=int,   default=32)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--max-nodes",       type=int,   default=500,
                    help="Drop graphs with more nodes than this (default 500); "
                         "protects against outliers like luaV_execute (6123 nodes)")
    ap.add_argument("--pretrain-only",   action="store_true",
                    help="Run Phase 1 only")
    ap.add_argument("--finetune-only",   action="store_true",
                    help="Skip Phase 1; load existing pretrain ckpt")
    ap.add_argument("--pretrain-ckpt",   type=str,
                    default="model_juliet_pretrain.pt",
                    help="Phase 1 checkpoint (shared with §27/§28)")
    ap.add_argument("--checkpoint",      type=str,
                    default="model_slice_pdg_v9.pt",
                    help="Final §30 checkpoint")
    args = ap.parse_args()

    if args.pretrain_only and args.finetune_only:
        print("ERROR: --pretrain-only and --finetune-only are mutually exclusive.")
        sys.exit(1)

    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n§30 Juliet positives + clean real-C negatives")
    print(f"Device: {device}")

    juliet_train = DATA / "train_juliet_graphs.pkl"
    juliet_valid = DATA / "valid_juliet_graphs.pkl"
    clean_train  = DATA / "train_clean_neg_graphs.pkl"
    clean_valid  = DATA / "valid_clean_neg_graphs.pkl"

    pretrain_ckpt = Path(args.pretrain_ckpt)
    final_ckpt    = Path(args.checkpoint)

    model = SlicePDGGNN_v7(VOCAB_SIZE, args.embed_dim, args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SlicePDGGNN_v7  params={n_params:,}")

    # =========================================================================
    # Phase 1 — Juliet-only BCE (reuse §27/§28 checkpoint if present)
    # =========================================================================
    if not args.finetune_only:
        for p in [juliet_train, juliet_valid]:
            if not p.exists():
                print(f"\nMissing: {p}")
                print("Run: python preprocess_juliet.py --workers 8")
                sys.exit(1)

        if pretrain_ckpt.exists():
            print(f"\n  Phase 1 checkpoint already exists: {pretrain_ckpt}")
            print(f"  Loading and skipping Phase 1 training.")
            model.load_state_dict(torch.load(pretrain_ckpt, map_location=device,
                                              weights_only=True))
        else:
            print("\n-- Loading Juliet graphs (Phase 1) --")
            j_train = load_graphs(juliet_train)
            j_valid = load_graphs(juliet_valid)
            nc = [d.x.shape[0] for d in j_train]
            print(f"  train={len(j_train)}  valid={len(j_valid)}")
            print(f"  Slice sizes: mean={statistics.mean(nc):.0f}  max={max(nc)}")

            run_phase(
                label="Phase 1 — Juliet BCE (structural prior)",
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
            return

    # =========================================================================
    # Phase 2 — Juliet bad + (Juliet good + real clean C) negatives
    # =========================================================================
    if not args.pretrain_only:
        for p in [juliet_train, juliet_valid, clean_train, clean_valid]:
            if not p.exists():
                if "clean" in p.name:
                    print(f"\nMissing: {p}")
                    print("Run: python preprocess_clean_negatives.py --workers 4")
                else:
                    print(f"\nMissing: {p}")
                    print("Run: python preprocess_juliet.py --workers 8")
                sys.exit(1)

        if args.finetune_only:
            if pretrain_ckpt.exists():
                print(f"\nLoading pretrained weights from {pretrain_ckpt} ...")
                model.load_state_dict(torch.load(pretrain_ckpt, map_location=device,
                                                  weights_only=True))
            else:
                print(f"WARNING: --finetune-only but {pretrain_ckpt} not found. "
                      f"Starting from random init.")

        print("\n-- Loading Phase 2 data --")
        print(f"  Juliet train:   {juliet_train.name}")
        print(f"  Juliet valid:   {juliet_valid.name}")
        print(f"  Clean neg train: {clean_train.name}")
        print(f"  Clean neg valid: {clean_valid.name}")

        d_train, d_valid = load_combined_splits(
            rng, juliet_train, juliet_valid, clean_train, clean_valid,
            max_nodes=args.max_nodes)

        n_vuln_tr  = sum(1 for d in d_train if d.y.item() == 1)
        n_vuln_va  = sum(1 for d in d_valid if d.y.item() == 1)
        print(f"\n  Phase 2 train: {len(d_train)}  "
              f"({n_vuln_tr} vuln / {len(d_train)-n_vuln_tr} benign)")
        print(f"  Phase 2 valid: {len(d_valid)}  "
              f"({n_vuln_va} vuln / {len(d_valid)-n_vuln_va} benign)")

        # Clean negative functions have no Devign ID — no test split to evaluate.
        # Validation accuracy is not the goal; scarnet ranking is.
        print(f"\n  NOTE: no Devign test set — §30 has no 'test accuracy' number.")
        print(f"  Evaluate with: python eval_all_models.py --scarnet")

        run_phase(
            label="Phase 2 — Juliet bad + real clean C negatives",
            model=model,
            train_data=d_train,
            valid_data=d_valid,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            batch_size=args.batch_size,
            device=device,
            checkpoint=final_ckpt,
        )

        print(f"\n{'='*55}")
        print(f"  §30 training complete")
        print(f"{'='*55}")
        print(f"  Checkpoint: {final_ckpt.resolve()}")
        print(f"")
        print(f"  Evaluate on scarnet:")
        print(f"    python eval_all_models.py --scarnet \\")
        print(f"      --answer-key ~/Downloads/SCAR/scarnet-answer-key.txt")
        print(f"")
        print(f"  What to look for:")
        print(f"    - Score spread: vuln functions 60-90%, benign functions <50%")
        print(f"    - session_new and dispatch (FPs in §27) should drop")
        print(f"    - scar_alloc_copy (miss in §27) should rise")
        print(f"    - handle_stats will still score low (no dangerous sink)")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
