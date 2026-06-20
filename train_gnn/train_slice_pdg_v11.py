#!/usr/bin/env python3
"""
train_slice_pdg_v11.py — §32 Juliet pretrain + Devign RankNet, with atoi sinks.

Two bugs were found during §31 analysis:

  1. eval_all_models.py used ir_to_graph_slice_pdg (x shape N×1) instead of
     ir_to_graph_slice_pdg_v7 (x shape N×3) for §27–§31, silently zeroing
     guard_class and is_external_input in every scarnet evaluation run.

  2. atoi/atol/atoll/strtol/strtoul/strtoll/strtoull/strtod were missing from
     DANGEROUS_SINKS in preprocess_juliet.py.  scar_atoi — a known-vulnerable
     CWE-190/191 integer conversion function — had no slice built during Juliet
     preprocessing.  The model never saw this vulnerability class as a positive
     during training, yet §28 still scored scar_atoi at 71.1% by generalising
     from other sink patterns.

Both bugs are now fixed.  §32 rebuilds Juliet graphs with atoi sinks included
and retrains Phase 2 (RankNet) from the existing model_juliet_pretrain.pt.
Phase 1 is unaffected — the pretrain checkpoint only uses opcode features.

Architecture: SlicePDGGNN_v7 — identical to §27/§28/§29/§30/§31.

Prerequisites:
    # Rebuild Juliet graphs with atoi/strtol in DANGEROUS_SINKS:
    python preprocess_juliet.py --workers 8

    # (model_juliet_pretrain.pt can be reused — Phase 1 is opcode-only)

Usage:
    python train_slice_pdg_v11.py --finetune-only   # recommended — reuse pretrain
    python train_slice_pdg_v11.py --help
"""

import argparse
import random
import statistics
import sys
from pathlib import Path

import torch

from train_slice_pdg_v7 import SlicePDGGNN_v7, load_graphs, run_phase
from train_slice_pdg_v8 import run_phase_ranknet

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
                    help="Phase 2 Devign RankNet epochs")
    ap.add_argument("--hidden",          type=int,   default=64)
    ap.add_argument("--embed-dim",       type=int,   default=128)
    ap.add_argument("--lr",              type=float, default=1e-3,
                    help="Phase 1 learning rate")
    ap.add_argument("--finetune-lr",     type=float, default=1e-4,
                    help="Phase 2 RankNet learning rate")
    ap.add_argument("--batch-size",      type=int,   default=32)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--pretrain-only",   action="store_true")
    ap.add_argument("--finetune-only",   action="store_true",
                    help="Skip Phase 1; load existing pretrain ckpt")
    ap.add_argument("--pretrain-ckpt",   type=str,
                    default="model_juliet_pretrain.pt",
                    help="Phase 1 checkpoint — shared with §27/§28/§29/§30/§31")
    ap.add_argument("--devign-train",    type=str,
                    default="data/train_graphs.pkl")
    ap.add_argument("--devign-valid",    type=str,
                    default="data/valid_graphs.pkl")
    ap.add_argument("--devign-test",     type=str,
                    default="data/test_graphs.pkl")
    ap.add_argument("--checkpoint",      type=str,
                    default="model_slice_pdg_v11.pt",
                    help="Final §32 checkpoint")
    args = ap.parse_args()

    if args.pretrain_only and args.finetune_only:
        print("ERROR: --pretrain-only and --finetune-only are mutually exclusive.")
        sys.exit(1)

    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n§32 Juliet pretrain (atoi sinks) + Devign RankNet fine-tune")
    print(f"Device: {device}")

    juliet_train = DATA / "train_juliet_graphs.pkl"
    juliet_valid = DATA / "valid_juliet_graphs.pkl"
    devign_train = Path(args.devign_train)
    devign_valid = Path(args.devign_valid)
    devign_test  = Path(args.devign_test)
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
                print(f"\nMissing: {p}\nRun: python preprocess_juliet.py --workers 8")
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
            print(f"  Feature cols: {j_train[0].x.shape[1]}  (expect 3, incl. guard+input)")

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
    # Phase 2 — Devign RankNet fine-tune
    # =========================================================================
    if not args.pretrain_only:
        for p in [devign_train, devign_valid]:
            if not p.exists():
                print(f"\nMissing: {p}\nRun: python preprocess_slice_pdg.py first")
                sys.exit(1)

        if args.finetune_only:
            if pretrain_ckpt.exists():
                print(f"\nLoading pretrained weights from {pretrain_ckpt} ...")
                model.load_state_dict(torch.load(pretrain_ckpt, map_location=device,
                                                  weights_only=True))
            else:
                print(f"WARNING: --finetune-only but {pretrain_ckpt} not found. "
                      f"Starting from random init.")

        print("\n-- Loading Devign graphs (Phase 2) --")
        d_train = load_graphs(devign_train)
        d_valid = load_graphs(devign_valid)
        n_vuln_tr = sum(1 for d in d_train if d.y.item() == 1)
        print(f"  train={len(d_train)}  ({n_vuln_tr} vuln / {len(d_train)-n_vuln_tr} benign)")
        print(f"  valid={len(d_valid)}")

        run_phase_ranknet(
            label="Phase 2 — Devign RankNet (§32)",
            model=model,
            train_data=d_train,
            valid_data=d_valid,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            batch_size=args.batch_size,
            device=device,
            checkpoint=final_ckpt,
        )

        # Devign test accuracy
        if devign_test.exists():
            import torch.nn.functional as F
            from torch_geometric.loader import DataLoader
            d_test  = load_graphs(devign_test)
            loader  = DataLoader(d_test, batch_size=64)
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    out   = torch.sigmoid(model(batch.x, batch.edge_index,
                                                batch.edge_type, batch.batch))
                    pred  = (out > 0.5).float()
                    correct += (pred.squeeze() == batch.y).sum().item()
            acc = correct / len(d_test)
            print(f"\n  Devign test accuracy: {acc:.2%}")

        print(f"\n{'='*55}")
        print(f"  §32 training complete")
        print(f"{'='*55}")
        print(f"  Checkpoint: {final_ckpt.resolve()}")
        print(f"")
        print(f"  Evaluate on scarnet:")
        print(f"    python eval_all_models.py --scarnet \\")
        print(f"      --answer-key ~/Downloads/SCAR/scarnet-answer-key.txt")
        print(f"")
        print(f"  What to look for vs §28:")
        print(f"    - scar_atoi (71.1% rank 6 in §28) should rise")
        print(f"    - scar_alloc_copy (68.3% rank 14 in §28) may cross into top 13")
        print(f"    - handle_stats should remain low (no dangerous sink)")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
