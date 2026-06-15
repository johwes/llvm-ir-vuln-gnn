#!/usr/bin/env python3
"""
train_triplet.py — Triplet Contrastive Learning on BigVul (vuln, fix) pairs.

Unlike SupCon (which collapsed on Devign because it pulled all vulnerable
functions together regardless of type), Triplet Loss uses the exact fix as
the guaranteed negative for each anchor:

    anchor   = vulnerable function IR graph
    negative = its exact patched counterpart (same commit, same function)
    positive = another vulnerable function, same CWE preferred

The gradient is unambiguous: push THIS function away from ITS patch.
The model cannot collapse to a generic vector because each negative is
structurally specific to its anchor.

Architecture: identical ContrastiveGNN encoder from train_contrastive.py.
Inference: k-NN against corpus of vuln embeddings (same as train_contrastive).

Usage:
    python train_triplet.py                          # 50 epochs, defaults
    python train_triplet.py --epochs 30 --margin 0.3
    python train_triplet.py --batch-size 64
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch_geometric.data import Data, Batch

HERE = Path(__file__).parent
DATA = HERE / "data"

sys.path.insert(0, str(HERE))
from train_contrastive import ContrastiveGNN, build_corpus, knn_accuracy

N_FEATURES = 45


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """Each item is a (vuln_Data, fix_Data, cwe_str) triple."""

    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        vuln = self._to_data(rec["vuln"])
        fix  = self._to_data(rec["fix"])
        return vuln, fix, rec.get("cwe", "CWE-unknown")

    @staticmethod
    def _to_data(g: dict) -> Data:
        x          = torch.tensor(g["x"],          dtype=torch.float)
        edge_index = torch.tensor(g["edge_index"],  dtype=torch.long)
        edge_type  = torch.tensor(g["edge_type"],   dtype=torch.long)
        if x.shape[0] > 1:
            x = (x - x.mean(0)) / (x.std(0) + 1e-8)
        return Data(x=x, edge_index=edge_index, edge_type=edge_type)


def pair_collate(batch):
    """Collate (vuln, fix, cwe) triples into two PyG batches + cwe list."""
    vulns, fixes, cwes = zip(*batch)
    return Batch.from_data_list(list(vulns)), Batch.from_data_list(list(fixes)), list(cwes)


def load_pairs(pkl_path: Path) -> list[dict]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Triplet loss
# ---------------------------------------------------------------------------

def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor,
                 negative: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """
    Cosine-similarity triplet loss on L2-normalised embeddings.

    anchor   — vuln embedding
    positive — same-CWE vuln embedding (or random vuln)
    negative — exact fix embedding for the anchor

    Loss = mean over active triplets of: max(0, sim(a,neg) - sim(a,pos) + margin)
    Only triplets where the loss > 0 contribute (semi-hard mining effect).
    """
    pos_sim = (anchor * positive).sum(dim=-1)   # cosine similarity
    neg_sim = (anchor * negative).sum(dim=-1)
    loss    = torch.clamp(neg_sim - pos_sim + margin, min=0.0)
    active  = loss > 0
    if not active.any():
        return loss.mean()
    return loss[active].mean()


# ---------------------------------------------------------------------------
# Positive mining: same-CWE vuln in batch, fall back to random
# ---------------------------------------------------------------------------

def mine_positives(vuln_emb: torch.Tensor, cwes: list[str]) -> torch.Tensor:
    """
    For each anchor i, find another vuln j with the same CWE.
    Fall back to the nearest other vuln (by cosine sim) if no same-CWE exists.
    Returns a (N, D) tensor of positive embeddings.
    """
    N = vuln_emb.shape[0]
    positives = torch.zeros_like(vuln_emb)

    cwe_to_idx: dict[str, list[int]] = {}
    for i, cwe in enumerate(cwes):
        cwe_to_idx.setdefault(cwe, []).append(i)

    # Cosine sim matrix for fallback
    sim = torch.mm(vuln_emb, vuln_emb.T)
    eye = torch.eye(N, dtype=torch.bool, device=vuln_emb.device)
    sim[eye] = -2.0  # exclude self

    for i, cwe in enumerate(cwes):
        candidates = [j for j in cwe_to_idx.get(cwe, []) if j != i]
        if candidates:
            j = candidates[torch.randint(len(candidates), (1,)).item()]
        else:
            j = sim[i].argmax().item()
        positives[i] = vuln_emb[j]

    return positives


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(model: ContrastiveGNN, loader, optimizer,
                device: torch.device, margin: float) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for vuln_batch, fix_batch, cwes in loader:
        vuln_batch = vuln_batch.to(device)
        fix_batch  = fix_batch.to(device)
        optimizer.zero_grad()

        # Forward: projected L2-normalised embeddings (training mode)
        vuln_emb = model(vuln_batch.x, vuln_batch.edge_index,
                         vuln_batch.edge_type, vuln_batch.batch)
        fix_emb  = model(fix_batch.x,  fix_batch.edge_index,
                         fix_batch.edge_type,  fix_batch.batch)

        positive_emb = mine_positives(vuln_emb.detach(), cwes)
        positive_emb = positive_emb.to(device)

        loss = triplet_loss(vuln_emb, positive_emb, fix_emb, margin)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def pair_similarity(model: ContrastiveGNN, loader,
                    device: torch.device) -> float:
    """Mean cosine similarity between (vuln, fix) pairs — should decrease."""
    model.eval()
    sims = []
    for vuln_batch, fix_batch, _ in loader:
        vuln_batch = vuln_batch.to(device)
        fix_batch  = fix_batch.to(device)
        vh = model.encode(vuln_batch.x, vuln_batch.edge_index,
                          vuln_batch.edge_type, vuln_batch.batch)
        fh = model.encode(fix_batch.x,  fix_batch.edge_index,
                          fix_batch.edge_type,  fix_batch.batch)
        vh = F.normalize(vh, dim=-1)
        fh = F.normalize(fh, dim=-1)
        sims.append((vh * fh).sum(dim=-1).cpu())
    return torch.cat(sims).mean().item()


# ---------------------------------------------------------------------------
# k-NN corpus loader (vuln graphs only — corpus = labelled vuln embeddings)
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_vuln_corpus(model: ContrastiveGNN, loader,
                      device: torch.device):
    """Build corpus from vuln embeddings only (label=1 for all)."""
    model.eval()
    embs = []
    for vuln_batch, _, _ in loader:
        vuln_batch = vuln_batch.to(device)
        h = model.encode(vuln_batch.x, vuln_batch.edge_index,
                         vuln_batch.edge_type, vuln_batch.batch)
        embs.append(h.cpu())
    embs = F.normalize(torch.cat(embs, dim=0), dim=-1)
    labs = torch.ones(embs.shape[0], dtype=torch.long)
    return embs, labs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=50)
    ap.add_argument("--hidden",     type=int,   default=64)
    ap.add_argument("--proj-dim",   type=int,   default=128)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int,   default=64)
    ap.add_argument("--margin",     type=float, default=0.3,
                    help="Triplet margin (cosine scale 0-2)")
    ap.add_argument("--k",          type=int,   default=5)
    ap.add_argument("--checkpoint", type=str,   default="encoder_triplet.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    for split in ["train", "valid", "test"]:
        pkl = DATA / f"bigvul_{split}_pairs.pkl"
        if not pkl.exists():
            print(f"Missing {pkl} — run preprocess_bigvul.py first.")
            sys.exit(1)

    print("Loading pairs ...")
    train_recs = load_pairs(DATA / "bigvul_train_pairs.pkl")
    valid_recs = load_pairs(DATA / "bigvul_valid_pairs.pkl")
    test_recs  = load_pairs(DATA / "bigvul_test_pairs.pkl")
    print(f"  train={len(train_recs):,}  valid={len(valid_recs):,}  "
          f"test={len(test_recs):,} pairs")

    cwes = [r.get("cwe", "CWE-unknown") for r in train_recs]
    from collections import Counter
    top = Counter(cwes).most_common(5)
    print(f"  Top CWEs in train: {top}\n")

    train_ds = PairDataset(train_recs)
    valid_ds = PairDataset(valid_recs)
    test_ds  = PairDataset(test_recs)

    train_loader  = TorchDataLoader(train_ds, batch_size=args.batch_size,
                                    shuffle=True,  collate_fn=pair_collate)
    valid_loader  = TorchDataLoader(valid_ds, batch_size=256,
                                    shuffle=False, collate_fn=pair_collate)
    test_loader   = TorchDataLoader(test_ds,  batch_size=256,
                                    shuffle=False, collate_fn=pair_collate)
    corpus_loader = TorchDataLoader(train_ds, batch_size=256,
                                    shuffle=False, collate_fn=pair_collate)

    model     = ContrastiveGNN(N_FEATURES, args.hidden, args.proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ContrastiveGNN(in={N_FEATURES}, hidden={args.hidden}, "
          f"proj={args.proj_dim})  params={n_params:,}")
    print(f"Loss:  Triplet  margin={args.margin}  batch={args.batch_size}  k={args.k}\n")

    best_val_acc = 0.0
    checkpoint   = Path(args.checkpoint)

    print(f"{'Epoch':>5}  {'Loss':>8}  {'Pair-Sim':>9}  {'Val k-NN':>9}  {'':>6}")
    print("-" * 48)

    for epoch in range(1, args.epochs + 1):
        loss    = train_epoch(model, train_loader, optimizer, device, args.margin)
        sim     = pair_similarity(model, valid_loader, device)
        scheduler.step()

        corpus_emb, corpus_lab = build_vuln_corpus(model, corpus_loader, device)

        # For k-NN eval, we need a mixed corpus (vuln + fix labelled)
        # Build fix embeddings labelled 0 and append
        fix_embs, fix_labs = [], []
        with torch.no_grad():
            model.eval()
            for _, fix_batch, _ in corpus_loader:
                fix_batch = fix_batch.to(device)
                h = model.encode(fix_batch.x, fix_batch.edge_index,
                                 fix_batch.edge_type, fix_batch.batch)
                fix_embs.append(h.cpu())
                fix_labs.append(torch.zeros(h.shape[0], dtype=torch.long))
        fix_embs = F.normalize(torch.cat(fix_embs, dim=0), dim=-1)
        fix_labs = torch.cat(fix_labs, dim=0)

        full_corpus_emb = torch.cat([corpus_emb, fix_embs], dim=0)
        full_corpus_lab = torch.cat([corpus_lab, fix_labs], dim=0)

        # Build val query loader: vuln=1, fix=0
        val_embs, val_labs = [], []
        with torch.no_grad():
            model.eval()
            for vuln_b, fix_b, _ in valid_loader:
                vuln_b = vuln_b.to(device)
                fix_b  = fix_b.to(device)
                vh = model.encode(vuln_b.x, vuln_b.edge_index,
                                  vuln_b.edge_type, vuln_b.batch)
                fh = model.encode(fix_b.x,  fix_b.edge_index,
                                  fix_b.edge_type,  fix_b.batch)
                val_embs += [F.normalize(vh, dim=-1).cpu(),
                             F.normalize(fh, dim=-1).cpu()]
                val_labs += [torch.ones(vh.shape[0],  dtype=torch.long),
                             torch.zeros(fh.shape[0], dtype=torch.long)]
        val_embs = torch.cat(val_embs, dim=0)
        val_labs = torch.cat(val_labs, dim=0)

        k = min(args.k, full_corpus_emb.shape[0])
        sims_mat = torch.mm(val_embs, full_corpus_emb.T)
        _, topk  = sims_mat.topk(k, dim=1)
        knn_lab  = full_corpus_lab[topk].float()
        preds    = (knn_lab.mean(dim=1) >= 0.5).long()
        val_acc  = (preds == val_labs).float().mean().item()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)
            marker = "<- best"

        print(f"{epoch:>5}  {loss:>8.4f}  {sim:>9.4f}  {val_acc:>9.2%}  {marker}")

    # Final test evaluation
    print(f"\nLoading best checkpoint ({checkpoint}) ...")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    test_embs, test_labs = [], []
    with torch.no_grad():
        model.eval()
        for vuln_b, fix_b, _ in test_loader:
            vuln_b = vuln_b.to(device)
            fix_b  = fix_b.to(device)
            vh = model.encode(vuln_b.x, vuln_b.edge_index,
                              vuln_b.edge_type, vuln_b.batch)
            fh = model.encode(fix_b.x,  fix_b.edge_index,
                              fix_b.edge_type,  fix_b.batch)
            test_embs += [F.normalize(vh, dim=-1).cpu(),
                          F.normalize(fh, dim=-1).cpu()]
            test_labs += [torch.ones(vh.shape[0],  dtype=torch.long),
                          torch.zeros(fh.shape[0], dtype=torch.long)]
    test_embs = torch.cat(test_embs, dim=0)
    test_labs = torch.cat(test_labs, dim=0)

    k = min(args.k, full_corpus_emb.shape[0])
    sims_mat = torch.mm(test_embs, full_corpus_emb.T)
    _, topk  = sims_mat.topk(k, dim=1)
    knn_lab  = full_corpus_lab[topk].float()
    preds    = (knn_lab.mean(dim=1) >= 0.5).long()
    test_acc = (preds == test_labs).float().mean().item()

    # Mean (vuln, fix) cosine similarity on test set
    test_pair_sim = pair_similarity(model, test_loader, device)

    print(f"\nTest k-NN accuracy (k={args.k}): {test_acc:.2%}")
    print(f"Mean (vuln, fix) cosine similarity: {test_pair_sim:.4f}  "
          f"(lower = better separation)")

    corpus_path = Path("corpus_triplet_embeddings.npz")
    np.savez(corpus_path,
             embeddings=full_corpus_emb.numpy(),
             labels=full_corpus_lab.numpy())
    print(f"Corpus saved -> {corpus_path.resolve()}\n")

    print("--- Results ---")
    print(f"  Devign GNN classifier (4d):              57.84%")
    print(f"  Devign SupCon k-NN (collapsed):          55.84%")
    print(f"  CodeBERT on Devign:                      63.43%")
    print(f"  BigVul Triplet k-NN (this run, k={args.k}):  {test_acc:.2%}")

    if test_acc >= 0.62:
        print("Triplet loss with exact (vuln, fix) pairs closes the gap — "
              "structural diff signal is real and learnable from IR")
    elif test_acc >= 0.58:
        print("Improvement over SupCon — exact negatives help, "
              "but block-level features still limit the ceiling")
    else:
        print("No improvement — structural ceiling applies to triplet loss too; "
              "instruction-level features or richer encodings needed")


if __name__ == "__main__":
    main()
