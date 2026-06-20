#!/usr/bin/env python3
"""
train_slice_pdg_v7.py — §27 Juliet pretraining → Devign fine-tune.

§27 experiment: Two-phase training to move the GNN from Philosophy 1
(distribution fingerprint) toward Philosophy 2 (structural pattern detection).

Phase 1 — Pretrain on Juliet (clean structural signal):
  ~100 k Juliet bad/good pairs, zero label noise, paired within CWE.
  The model learns "guarded sink vs unguarded sink", not "FFmpeg vs QEMU style".

Phase 2 — Fine-tune on Devign (real-world distribution):
  Same LLVM IR / same PDG slices as §12.  Crucially: the Devign preprocessor
  (preprocess_slice_pdg.py) now produces x shape (N, 1).  We re-run it here
  with the v7 multi-feature preprocessor, OR load pre-built Juliet graphs and
  use the existing §12 Devign graphs with a feature adapter.

  Architecture decision: the model has a learnable projection that maps
  x[:, 0] (opcode_id) → embed_dim.  Features x[:, 1] and x[:, 2] are
  concatenated as scalar floats after the embedding.  This lets us load
  Devign §12 graphs (x shape (N, 1)) and zero-pad the missing feature
  columns, so Phase 2 works with any existing Devign pickle.

New node features (§27 multi-feature x):
  col 0: opcode_id      (embedded via nn.Embedding, range 0-109)
  col 1: guard_class    (0=none, 1=bounds_check, 2=null_check)
  col 2: is_external_input (0 or 1)

  Devign graphs from §12 have x shape (N, 1): cols 1&2 default to 0.
  Juliet graphs from preprocess_juliet.py have x shape (N, 3).

Prerequisites:
    # Phase 1 data (run on machine with Juliet zip):
    python preprocess_juliet.py --workers 8

    # Phase 2 data — either existing §12 Devign pickle (auto-detected):
    #   data/train_slice_pdg_graphs.pkl   (§12, x shape (N,1))
    # OR re-preprocess with multi-feature preprocessor:
    #   python preprocess_slice_pdg.py  (with v7 features)

Usage:
    python train_slice_pdg_v7.py --help
    python train_slice_pdg_v7.py --pretrain-epochs 20 --finetune-epochs 30
    python train_slice_pdg_v7.py --pretrain-only           # phase 1 only
    python train_slice_pdg_v7.py --finetune-only           # skip phase 1
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

HERE  = Path(__file__).parent
DATA  = HERE / "data"

VOCAB_SIZE  = 110   # LLVM opcode vocab (shared with §12)
N_SCALAR    = 2     # guard_class, is_external_input

# ---------------------------------------------------------------------------
# Model — multi-feature RGCN
# ---------------------------------------------------------------------------

class SlicePDGGNN_v7(nn.Module):
    """
    Same RGCN backbone as §12 but with multi-feature input.

    Input:
      x[:, 0]   opcode_id  → nn.Embedding(vocab, embed_dim)
      x[:, 1:]  scalar features (guard_class, is_external_input) → linear proj

    Fusion: embed + proj are summed into a single hidden vector before RGCN.
    """

    def __init__(self,
                 vocab:     int = VOCAB_SIZE,
                 embed_dim: int = 128,
                 hidden:    int = 64,
                 n_scalar:  int = N_SCALAR):
        super().__init__()
        self.embed     = nn.Embedding(vocab, embed_dim)
        self.scalar_proj = nn.Linear(n_scalar, embed_dim) if n_scalar > 0 else None
        self.conv1     = RGCNConv(embed_dim, hidden, num_relations=3)
        self.conv2     = RGCNConv(hidden,    hidden, num_relations=3)
        gate_nn        = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )
        self.pool      = AttentionalAggregation(gate_nn=gate_nn)
        self.lin       = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_type, batch):
        # x shape: (N, 1) or (N, 3)
        opcode_ids = x[:, 0].clamp(0, VOCAB_SIZE - 1)
        h = self.embed(opcode_ids)

        if self.scalar_proj is not None and x.shape[1] > 1:
            scalars = x[:, 1:].float()
            h = h + self.scalar_proj(scalars)

        h = F.relu(self.conv1(h, edge_index, edge_type))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_type))
        h = self.pool(h, batch)
        return self.lin(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Data loading — handles (N,1) and (N,3) x tensors
# ---------------------------------------------------------------------------

def load_graphs(pkl_path: Path, zero_pad_to: int = 3) -> list[Data]:
    """Load a graph pickle and ensure x has exactly zero_pad_to columns."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    dataset = []
    for g in raw:
        x = torch.tensor(g["x"], dtype=torch.long)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.shape[1] < zero_pad_to:
            pad = torch.zeros(x.shape[0], zero_pad_to - x.shape[1], dtype=torch.long)
            x = torch.cat([x, pad], dim=1)
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


def _pos_weight(dataset: list[Data], device) -> torch.Tensor:
    n_vuln  = sum(1 for d in dataset if d.y.item() == 1)
    n_fixed = len(dataset) - n_vuln
    ratio   = n_fixed / n_vuln if n_vuln > 0 else 1.0
    return torch.tensor([ratio]).to(device)


def run_phase(label: str,
              model, train_data, valid_data,
              epochs: int, lr: float, batch_size: int,
              device, checkpoint: Path,
              lr_step: int = 10, lr_gamma: float = 0.5) -> float:
    """Train for `epochs` epochs, save best checkpoint, return best val acc."""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    n_vuln  = sum(1 for d in train_data if d.y.item() == 1)
    n_fixed = len(train_data) - n_vuln
    ratio   = n_fixed / n_vuln if n_vuln > 0 else 1.0
    pw      = torch.tensor([ratio]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=lr_step,
                                                  gamma=lr_gamma)

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  train={len(train_data)}  valid={len(valid_data)}")
    print(f"  class balance: {n_vuln} vuln / {n_fixed} benign  (pos_weight={ratio:.2f})")
    print(f"{'─'*55}")
    print(f"{'Epoch':>5}  {'Loss':>8}  {'Val Acc':>8}  {'':>6}")
    print("-" * 35)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        loss    = train_epoch(model, train_loader, optimizer, device, pw)
        val_acc = evaluate(model, valid_loader, device)
        scheduler.step()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)
            marker = "<- best"

        print(f"{epoch:>5}  {loss:>8.4f}  {val_acc:>8.2%}  {marker}")

    print(f"\n  Best val acc: {best_val_acc:.2%}  (checkpoint: {checkpoint})")
    return best_val_acc


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
                    help="Epochs for Juliet pretraining phase")
    ap.add_argument("--finetune-epochs", type=int,   default=30,
                    help="Epochs for Devign fine-tune phase")
    ap.add_argument("--hidden",          type=int,   default=64)
    ap.add_argument("--embed-dim",       type=int,   default=128)
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--finetune-lr",     type=float, default=3e-4,
                    help="Lower LR for fine-tune phase (default: 3e-4)")
    ap.add_argument("--batch-size",      type=int,   default=32)
    ap.add_argument("--pretrain-only",   action="store_true",
                    help="Run Phase 1 only; skip Devign fine-tune")
    ap.add_argument("--finetune-only",   action="store_true",
                    help="Skip Phase 1; load pretrained checkpoint and fine-tune")
    ap.add_argument("--pretrain-ckpt",   type=str,
                    default="model_juliet_pretrain.pt",
                    help="Checkpoint file for Juliet pretrained weights")
    ap.add_argument("--checkpoint",      type=str,
                    default="model_slice_pdg_v7.pt",
                    help="Final checkpoint after fine-tune (§27 model)")
    ap.add_argument("--devign-train",    type=str,   default=None,
                    help="Devign train pkl (auto-detected if not set)")
    ap.add_argument("--devign-valid",    type=str,   default=None)
    ap.add_argument("--devign-test",     type=str,   default=None)
    args = ap.parse_args()

    if args.pretrain_only and args.finetune_only:
        print("ERROR: --pretrain-only and --finetune-only are mutually exclusive.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n§27 Juliet-pretrain → Devign-finetune")
    print(f"Device: {device}")

    # -- Phase 1 data paths ---------------------------------------------------
    juliet_train = DATA / "train_juliet_graphs.pkl"
    juliet_valid = DATA / "valid_juliet_graphs.pkl"

    # -- Phase 2 data paths: prefer v7 pkl, fall back to §12 pkl -------------
    def _devign_path(split: str, override: str | None) -> Path:
        if override:
            return Path(override)
        # prefer multi-feature version produced by a re-run of preprocess
        candidates = [
            DATA / f"{split}_slice_pdg_v7_graphs.pkl",
            DATA / f"{split}_slice_pdg_graphs.pkl",       # §12 fallback
        ]
        for p in candidates:
            if p.exists():
                return p
        return candidates[-1]  # will fail with clear message below

    devign_train_path = _devign_path("train",  args.devign_train)
    devign_valid_path = _devign_path("valid",  args.devign_valid)
    devign_test_path  = _devign_path("test",   args.devign_test)

    pretrain_ckpt = Path(args.pretrain_ckpt)
    final_ckpt    = Path(args.checkpoint)

    n_params_msg  = None
    model = SlicePDGGNN_v7(VOCAB_SIZE, args.embed_dim, args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SlicePDGGNN_v7(vocab={VOCAB_SIZE}, embed={args.embed_dim}, "
          f"hidden={args.hidden}, n_scalar={N_SCALAR})  params={n_params:,}")

    # =========================================================================
    # Phase 1 — Pretrain on Juliet
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
        print(f"  train={len(j_train)}  valid={len(j_valid)}")

        nc = [d.x.shape[0] for d in j_train]
        print(f"  Slice sizes: mean={statistics.mean(nc):.0f}  "
              f"median={statistics.median(nc):.0f}  max={max(nc)}")
        print(f"  Feature cols: {j_train[0].x.shape[1]}  (expect 3)")

        run_phase(
            label="Phase 1 — Juliet pretraining",
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
    # Phase 2 — Fine-tune on Devign
    # =========================================================================
    if not args.pretrain_only:
        for p in [devign_train_path, devign_valid_path, devign_test_path]:
            if not p.exists():
                print(f"\nMissing: {p}")
                print("Run: python preprocess_slice_pdg.py")
                sys.exit(1)

        # Load pretrained weights if they exist (Phase 1 was already run)
        if pretrain_ckpt.exists() and not args.finetune_only:
            print(f"\nLoading Phase 1 weights from {pretrain_ckpt} ...")
            model.load_state_dict(torch.load(pretrain_ckpt, map_location=device,
                                              weights_only=True))
        elif args.finetune_only and pretrain_ckpt.exists():
            print(f"\nLoading pretrained weights from {pretrain_ckpt} ...")
            model.load_state_dict(torch.load(pretrain_ckpt, map_location=device,
                                              weights_only=True))
        elif args.finetune_only:
            print(f"WARNING: --finetune-only but {pretrain_ckpt} not found.")
            print(f"         Starting fine-tune from random init.")

        print(f"\n-- Loading Devign graphs --")
        print(f"  train: {devign_train_path.name}")
        print(f"  valid: {devign_valid_path.name}")
        print(f"  test:  {devign_test_path.name}")
        d_train = load_graphs(devign_train_path)
        d_valid = load_graphs(devign_valid_path)
        d_test  = load_graphs(devign_test_path)
        print(f"  train={len(d_train)}  valid={len(d_valid)}  test={len(d_test)}")

        feat_cols = d_train[0].x.shape[1] if d_train else 1
        print(f"  Feature cols: {feat_cols}"
              f"  {'(§12 compat — cols 1&2 zero-padded)' if feat_cols == 1 else '(multi-feature)'}")

        best_val = run_phase(
            label="Phase 2 — Devign fine-tune",
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
        print(f"  §27 results summary")
        print(f"{'='*55}")
        print(f"  Phase 1 pretrain:   Juliet  (clean structural signal)")
        print(f"  Phase 2 fine-tune:  Devign  ({devign_train_path.name})")
        print(f"  Devign test acc:    {test_acc:.2%}")
        print(f"")
        print(f"  Baseline comparisons:")
        print(f"    §12  PDG slice, Devign only:         56.48%  (target to beat)")
        print(f"    §25  PDG slice, PrimeVul clang:      TBD")
        print(f"    §27  Juliet pretrain + Devign FT:    {test_acc:.2%}  <-- this run")
        print(f"")
        print(f"  Note: Devign test accuracy is NOT the real metric.")
        print(f"  ~10-20% label noise creates a floor. Run eval_all_models.py")
        print(f"  on scarnet to measure ranking quality (NDCG, MRR).")
        print(f"")
        print(f"  scarnet eval:")
        print(f"    python eval_all_models.py --scarnet \\")
        print(f"      --model-dir . \\")
        print(f"      --model-files model_slice_pdg_v7.pt \\")
        print(f"      --answer-key ~/Downloads/SCAR/scarnet-answer-key.txt")
        print(f"{'='*55}")
        print(f"  Checkpoint: {final_ckpt.resolve()}")


if __name__ == "__main__":
    main()
