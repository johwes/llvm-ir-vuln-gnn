#!/usr/bin/env python3
"""
train_slice_pdg_v10.py — §31 Juliet positives + domain-matched clean negatives.

§30 showed that the approach (Juliet pos + clean real-C neg) is correct in
principle — training converged cleanly at 97.79% val accuracy — but the negative
corpus caused a domain-shift failure.  Lua (61% of negatives) taught the model
"interpreter C = clean", so server-style vulnerable functions (session_frag,
session_consume_frag, scar_atoi, parse_batch) scored near 0%.

§31 fixes this with domain-matched negatives:
  - libcurl replaces lua as the primary negative source
  - libcurl is network/protocol C: socket handling, buffer mgmt, HTTP parsing,
    session management — structurally closest to scarnet among all available
    open-source C libraries
  - All sources are heavily audited; any unguarded pattern has been fixed

Positives:  Juliet bad functions         (unguarded sinks, zero label noise)
Negatives:  Juliet good functions
          + zlib, musl, libcurl, lz4, cjson, libuv   (domain-diverse clean C)

Two-phase training:
  Phase 1 — Juliet bad vs Juliet good  (reuses model_juliet_pretrain.pt if present)
  Phase 2 — Juliet bad vs (Juliet good + domain-matched clean C)
             Output: model_slice_pdg_v10.pt

Architecture: SlicePDGGNN_v7 — identical to §27/§28/§29/§30.

Prerequisites:
    # Juliet graphs (reuse from §27–§30 if already built):
    python preprocess_juliet.py --workers 8

    # Domain-matched clean negatives (§31: libcurl replaces lua):
    python preprocess_clean_negatives.py \\
        --sources zlib,musl,libcurl,lz4,cjson,libuv \\
        --workers 4

Usage:
    python train_slice_pdg_v10.py --help
    python train_slice_pdg_v10.py --finetune-only   # skip Phase 1 if ckpt exists
    python train_slice_pdg_v10.py --finetune-epochs 40
"""

import argparse
import pickle
import random
import statistics
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

from train_slice_pdg_v7 import SlicePDGGNN_v7, load_graphs, run_phase
from train_slice_pdg_v9 import load_combined_splits

HERE = Path(__file__).parent
DATA = HERE / "data"

VOCAB_SIZE = 110
N_SCALAR   = 2


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    ap.add_argument("--pretrain-epochs", type=int,   default=20,
                    help="Phase 1 Juliet-only BCE epochs (skipped if ckpt exists)")
    ap.add_argument("--finetune-epochs", type=int,   default=40,
                    help="Phase 2 Juliet+domain-matched BCE epochs")
    ap.add_argument("--hidden",          type=int,   default=64)
    ap.add_argument("--embed-dim",       type=int,   default=128)
    ap.add_argument("--lr",              type=float, default=1e-3,
                    help="Phase 1 learning rate")
    ap.add_argument("--finetune-lr",     type=float, default=3e-4,
                    help="Phase 2 learning rate")
    ap.add_argument("--batch-size",      type=int,   default=32)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--max-nodes",       type=int,   default=500,
                    help="Drop graphs with more nodes than this (default 500)")
    ap.add_argument("--pretrain-only",   action="store_true",
                    help="Run Phase 1 only")
    ap.add_argument("--finetune-only",   action="store_true",
                    help="Skip Phase 1; load existing pretrain ckpt")
    ap.add_argument("--pretrain-ckpt",   type=str,
                    default="model_juliet_pretrain.pt",
                    help="Phase 1 checkpoint (shared with §27–§30)")
    ap.add_argument("--checkpoint",      type=str,
                    default="model_slice_pdg_v10.pt",
                    help="Final §31 checkpoint")
    args = ap.parse_args()

    if args.pretrain_only and args.finetune_only:
        print("ERROR: --pretrain-only and --finetune-only are mutually exclusive.")
        sys.exit(1)

    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n§31 Juliet positives + domain-matched clean negatives (libcurl replaces lua)")
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
    # Phase 1 — Juliet-only BCE (reuse §27–§30 checkpoint if present)
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
    # Phase 2 — Juliet bad + domain-matched clean C negatives
    # =========================================================================
    if not args.pretrain_only:
        for p in [juliet_train, juliet_valid, clean_train, clean_valid]:
            if not p.exists():
                if "clean" in p.name:
                    print(f"\nMissing: {p}")
                    print("Run: python preprocess_clean_negatives.py \\")
                    print("         --sources zlib,musl,libcurl,lz4,cjson,libuv \\")
                    print("         --workers 4")
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
        print(f"  Juliet train:    {juliet_train.name}")
        print(f"  Juliet valid:    {juliet_valid.name}")
        print(f"  Clean neg train: {clean_train.name}")
        print(f"  Clean neg valid: {clean_valid.name}")
        print(f"  NOTE: clean_neg pkl must have been built with "
              f"--sources zlib,musl,libcurl,lz4,cjson,libuv (no lua)")

        d_train, d_valid = load_combined_splits(
            rng, juliet_train, juliet_valid, clean_train, clean_valid,
            max_nodes=args.max_nodes)

        n_vuln_tr  = sum(1 for d in d_train if d.y.item() == 1)
        n_vuln_va  = sum(1 for d in d_valid if d.y.item() == 1)
        print(f"\n  Phase 2 train: {len(d_train)}  "
              f"({n_vuln_tr} vuln / {len(d_train)-n_vuln_tr} benign)")
        print(f"  Phase 2 valid: {len(d_valid)}  "
              f"({n_vuln_va} vuln / {len(d_valid)-n_vuln_va} benign)")

        print(f"\n  NOTE: no Devign test set — §31 has no 'test accuracy' number.")
        print(f"  Evaluate with: python eval_all_models.py --scarnet")

        run_phase(
            label="Phase 2 — Juliet bad + domain-matched clean negatives (§31)",
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
        print(f"  §31 training complete")
        print(f"{'='*55}")
        print(f"  Checkpoint: {final_ckpt.resolve()}")
        print(f"")
        print(f"  Evaluate on scarnet:")
        print(f"    python eval_all_models.py --scarnet \\")
        print(f"      --answer-key ~/Downloads/SCAR/scarnet-answer-key.txt")
        print(f"")
        print(f"  What to look for vs §30:")
        print(f"    - session_frag, session_consume_frag, scar_atoi, parse_batch")
        print(f"      should recover from 0.0% to 40–70% range")
        print(f"    - handle_del should remain near 99% (clear-sink pattern unchanged)")
        print(f"    - session_new may remain a FP or drop (structurally complex)")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
