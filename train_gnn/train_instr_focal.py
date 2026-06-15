#!/usr/bin/env python3
"""
train_instr_focal.py — §10b: Focal Contrastive Loss + SAGPooling on BigVul.

Two architectural changes over §8 (train_instr_triplet.py):

1. Focal Contrastive Loss (FCL) replaces triplet loss.
   The (1-p_ij)^gamma modifier amplifies the gradient exactly in the collapse
   regime (p_ij -> 1/n_batch = uniform), where standard triplet loss vanishes.
   This is the mathematical fix for the pair-sim 0.9995 collapse seen in §8.

2. SAGPooling replaces AttentionalAggregation.
   Keeps the top 25% of nodes by learned attention score, then mean-pools.
   Concentrates the embedding on high-degree vulnerability sinks rather than
   averaging across ~400 mostly-boilerplate nodes (0.5% patch signal).

Both changes together are the §10b experiment.
Baselines: §8 k-NN 48.39%, pair-sim 0.9984->0.9995 (soft collapse).

Architecture:
  Embedding:  nn.Embedding(110, 128, padding_idx=79)
  Encoder:    RGCNConv(128->64, 3 rels) x 2 + SAGPooling(ratio=0.25) + global_mean_pool
  Proj head:  Linear(64->128) -> ReLU -> Linear(128->128) -> L2-norm  (training only)
  Inference:  encoder output, cosine k-NN

Loss: Focal Contrastive (FCL)
  In-batch positives: vuln pairs (label=1) and fix pairs (label=0)
  No explicit positive mining — all same-class in-batch pairs are positives.
  tau:   temperature (default 0.07)
  gamma: focusing parameter (default 2.0)

Dataset: BigVul instruction-level pairs from preprocess_instr_bigvul.py
  data/bigvul_{train,valid,test}_instr_pairs.pkl

Usage:
    python train_instr_focal.py                          # 50 epochs, defaults
    python train_instr_focal.py --epochs 30 --hidden 64  # faster first run
    python train_instr_focal.py --tau 0.1 --gamma 1.0    # softer focal
"""

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import RGCNConv, SAGPooling, global_mean_pool

HERE = Path(__file__).parent
DATA = HERE / "data"

VOCAB   = 110
PAD_IDX = 79


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class InstructionContrastiveGNN(nn.Module):
    """
    Instruction-level GNN encoder + projection head for FCL training.

    Changes from §8 (train_instr_triplet.py):
      - SAGPooling(ratio=pool_ratio) replaces AttentionalAggregation
      - global_mean_pool over retained nodes replaces gate-network aggregation

    encode() returns the hidden-dim graph embedding (used at inference).
    forward() returns the L2-normalised projected embedding (used for loss).
    """

    def __init__(self, vocab: int = VOCAB, embed_dim: int = 128,
                 hidden: int = 64, proj_dim: int = 128,
                 pool_ratio: float = 0.25):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=PAD_IDX)
        self.conv1 = RGCNConv(embed_dim, hidden, num_relations=3)
        self.conv2 = RGCNConv(hidden,    hidden, num_relations=3)
        self.pool  = SAGPooling(hidden, ratio=pool_ratio)
        self.proj  = nn.Sequential(
            nn.Linear(hidden, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def encode(self, x, edge_index, edge_type, batch) -> torch.Tensor:
        """Returns hidden-dim graph embedding. Use at inference time."""
        h = self.embed(x.squeeze(-1))                          # (N,) -> (N, embed_dim)
        h = F.relu(self.conv1(h, edge_index, edge_type))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_type))
        # SAGPooling uses edge_index for its internal GCN scorer.
        # Edge type is already encoded in h by the RGCN layers above.
        h, edge_index_p, _, batch_p, _, _ = self.pool(h, edge_index, batch=batch)
        return global_mean_pool(h, batch_p)                    # (B, hidden)

    def forward(self, x, edge_index, edge_type, batch) -> torch.Tensor:
        """Returns L2-normalised projected embedding. Used for FCL."""
        h = self.encode(x, edge_index, edge_type, batch)
        return F.normalize(self.proj(h), dim=-1)               # (B, proj_dim)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """Each item is a (vuln_Data, fix_Data, cwe_str) triple."""

    def __init__(self, records: list):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec  = self.records[idx]
        vuln = self._to_data(rec["vuln"])
        fix  = self._to_data(rec["fix"])
        return vuln, fix, rec.get("cwe", "CWE-unknown")

    @staticmethod
    def _to_data(g: dict) -> Data:
        x          = torch.tensor(g["x"],         dtype=torch.long)
        edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
        edge_type  = torch.tensor(g["edge_type"],  dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_type=edge_type)


def pair_collate(batch):
    """Collate (vuln, fix, cwe) triples into two PyG batches + cwe list."""
    vulns, fixes, cwes = zip(*batch)
    return Batch.from_data_list(list(vulns)), Batch.from_data_list(list(fixes)), list(cwes)


def load_pairs(pkl_path: Path) -> list:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Focal Contrastive Loss
# ---------------------------------------------------------------------------

def focal_contrastive_loss(vuln_emb: torch.Tensor, fix_emb: torch.Tensor,
                            tau: float = 0.07, gamma: float = 2.0) -> torch.Tensor:
    """
    In-batch Focal Contrastive Loss (FCL).

    Stacks [vuln_emb | fix_emb] into a (2B, D) matrix.
    Positives for each anchor: all same-class embeddings in the batch.
      - vuln anchors: all other vulns in batch
      - fix  anchors: all other fixes  in batch

    The focal modifier (1-p_ij)^gamma is detached so it scales gradients
    without appearing in the backward graph itself.  When p_ij -> 1/n
    (uniform, collapse regime) the modifier approaches 1 and full gradient
    flows.  When p_ij is high (well-separated positive), modifier -> 0 and
    the pair is de-emphasised (easy pairs receive less gradient).

    Args:
        vuln_emb: (B, D) L2-normalised projected embeddings of vuln graphs
        fix_emb:  (B, D) L2-normalised projected embeddings of fix graphs
        tau:      temperature (default 0.07)
        gamma:    focusing parameter (default 2.0)

    Returns scalar loss.  Returns 0.0 if batch has < 2 same-class pairs.
    """
    B   = vuln_emb.shape[0]
    emb = torch.cat([vuln_emb, fix_emb], dim=0)           # (2B, D)
    labels = torch.cat([
        torch.ones(B,  dtype=torch.long),
        torch.zeros(B, dtype=torch.long),
    ], dim=0).to(emb.device)

    n   = 2 * B
    sim = torch.mm(emb, emb.T) / tau                       # (2B, 2B)
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)

    # pos_mask[i, j] = True iff j is a different item with the same label
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye  # (2B, 2B)

    n_pos = pos_mask.sum(dim=1)
    if n_pos.max().item() == 0:
        return emb.sum() * 0.0                             # no positives: stable zero

    # log p_ij = sim[i,j]/tau - logsumexp_{k!=i} sim[i,k]/tau
    log_denom = torch.logsumexp(sim.masked_fill(eye, float('-inf')), dim=1, keepdim=True)
    log_p     = sim - log_denom                            # (2B, 2B)

    # Focal weight — detached: scales gradient, not part of backward graph
    p     = log_p.exp().detach()                           # (2B, 2B)
    focal = (1.0 - p).pow(gamma)                           # (2B, 2B)

    n_pos_clamped = n_pos.float().clamp(min=1.0)
    loss = -(focal * log_p * pos_mask).sum(dim=1) / n_pos_clamped
    return loss.mean()


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(model: InstructionContrastiveGNN, loader,
                optimizer, device: torch.device,
                tau: float, gamma: float) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for vuln_batch, fix_batch, _ in loader:
        vuln_batch = vuln_batch.to(device)
        fix_batch  = fix_batch.to(device)
        optimizer.zero_grad()

        vuln_emb = model(vuln_batch.x, vuln_batch.edge_index,
                         vuln_batch.edge_type, vuln_batch.batch)
        fix_emb  = model(fix_batch.x,  fix_batch.edge_index,
                         fix_batch.edge_type,  fix_batch.batch)

        loss = focal_contrastive_loss(vuln_emb, fix_emb, tau=tau, gamma=gamma)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def pair_similarity(model: InstructionContrastiveGNN, loader,
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
# k-NN evaluation (mixed vuln+fix corpus)
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_mixed_corpus(model: InstructionContrastiveGNN, loader,
                       device: torch.device):
    """Build a (embeddings, labels) corpus from training pairs."""
    model.eval()
    vuln_embs, fix_embs = [], []
    for vuln_batch, fix_batch, _ in loader:
        vuln_batch = vuln_batch.to(device)
        fix_batch  = fix_batch.to(device)
        vh = model.encode(vuln_batch.x, vuln_batch.edge_index,
                          vuln_batch.edge_type, vuln_batch.batch)
        fh = model.encode(fix_batch.x,  fix_batch.edge_index,
                          fix_batch.edge_type,  fix_batch.batch)
        vuln_embs.append(vh.cpu())
        fix_embs.append(fh.cpu())

    vuln_embs = F.normalize(torch.cat(vuln_embs, dim=0), dim=-1)
    fix_embs  = F.normalize(torch.cat(fix_embs,  dim=0), dim=-1)
    embs      = torch.cat([vuln_embs, fix_embs], dim=0)
    labs      = torch.cat([
        torch.ones(vuln_embs.shape[0],  dtype=torch.long),
        torch.zeros(fix_embs.shape[0],  dtype=torch.long),
    ], dim=0)
    return embs, labs


@torch.no_grad()
def knn_accuracy(model: InstructionContrastiveGNN, corpus_emb: torch.Tensor,
                 corpus_lab: torch.Tensor, eval_loader,
                 device: torch.device, k: int) -> float:
    """k-NN accuracy on eval split, querying against mixed corpus."""
    model.eval()
    query_embs, query_labs = [], []
    for vuln_batch, fix_batch, _ in eval_loader:
        vuln_batch = vuln_batch.to(device)
        fix_batch  = fix_batch.to(device)
        vh = model.encode(vuln_batch.x, vuln_batch.edge_index,
                          vuln_batch.edge_type, vuln_batch.batch)
        fh = model.encode(fix_batch.x,  fix_batch.edge_index,
                          fix_batch.edge_type,  fix_batch.batch)
        query_embs += [F.normalize(vh, dim=-1).cpu(),
                       F.normalize(fh, dim=-1).cpu()]
        query_labs += [torch.ones(vh.shape[0],  dtype=torch.long),
                       torch.zeros(fh.shape[0], dtype=torch.long)]

    query_embs = torch.cat(query_embs, dim=0)
    query_labs = torch.cat(query_labs, dim=0)

    k_eff    = min(k, corpus_emb.shape[0])
    sims_mat = torch.mm(query_embs, corpus_emb.T)
    _, topk  = sims_mat.topk(k_eff, dim=1)
    knn_lab  = corpus_lab[topk].float()
    preds    = (knn_lab.mean(dim=1) >= 0.5).long()
    return (preds == query_labs).float().mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=50)
    ap.add_argument("--hidden",     type=int,   default=64)
    ap.add_argument("--embed-dim",  type=int,   default=128)
    ap.add_argument("--proj-dim",   type=int,   default=128)
    ap.add_argument("--pool-ratio", type=float, default=0.25,
                    help="SAGPooling retention ratio (0.25 = keep top 25%% of nodes)")
    ap.add_argument("--tau",        type=float, default=0.07,
                    help="FCL temperature (lower = sharper; 0.07 is SupCon standard)")
    ap.add_argument("--gamma",      type=float, default=2.0,
                    help="FCL focusing parameter (higher = more focus on hard pairs)")
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int,   default=64)
    ap.add_argument("--k",          type=int,   default=5)
    ap.add_argument("--checkpoint", type=str,   default="encoder_instr_focal.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    for split in ["train", "valid", "test"]:
        pkl = DATA / f"bigvul_{split}_instr_pairs.pkl"
        if not pkl.exists():
            print(f"Missing {pkl} -- run preprocess_instr_bigvul.py first.")
            sys.exit(1)

    print("Loading pairs ...")
    train_recs = load_pairs(DATA / "bigvul_train_instr_pairs.pkl")
    valid_recs = load_pairs(DATA / "bigvul_valid_instr_pairs.pkl")
    test_recs  = load_pairs(DATA / "bigvul_test_instr_pairs.pkl")
    print(f"  train={len(train_recs):,}  valid={len(valid_recs):,}  "
          f"test={len(test_recs):,} pairs")

    cwes = [r.get("cwe", "CWE-unknown") for r in train_recs]
    top  = Counter(cwes).most_common(5)
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

    model = InstructionContrastiveGNN(
        vocab=VOCAB, embed_dim=args.embed_dim,
        hidden=args.hidden, proj_dim=args.proj_dim,
        pool_ratio=args.pool_ratio,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: InstructionContrastiveGNN(vocab={VOCAB}, embed={args.embed_dim}, "
          f"hidden={args.hidden}, proj={args.proj_dim}, pool_ratio={args.pool_ratio})  "
          f"params={n_params:,}")
    print(f"Loss:  FCL  tau={args.tau}  gamma={args.gamma}  "
          f"batch={args.batch_size}  k={args.k}\n")

    best_val_acc = 0.0
    checkpoint   = Path(args.checkpoint)

    print(f"{'Epoch':>5}  {'Loss':>8}  {'Pair-Sim':>9}  {'Val k-NN':>9}  {'':>6}")
    print("-" * 48)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device,
                           tau=args.tau, gamma=args.gamma)
        sim  = pair_similarity(model, valid_loader, device)
        scheduler.step()

        corpus_emb, corpus_lab = build_mixed_corpus(model, corpus_loader, device)
        val_acc = knn_accuracy(model, corpus_emb, corpus_lab,
                               valid_loader, device, args.k)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)
            marker = "<- best"

        print(f"{epoch:>5}  {loss:>8.4f}  {sim:>9.4f}  {val_acc:>9.2%}  {marker}")

    # Final test evaluation
    print(f"\nLoading best checkpoint ({checkpoint}) ...")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    corpus_emb, corpus_lab = build_mixed_corpus(model, corpus_loader, device)
    test_acc      = knn_accuracy(model, corpus_emb, corpus_lab,
                                 test_loader, device, args.k)
    test_pair_sim = pair_similarity(model, test_loader, device)

    print(f"\nTest k-NN accuracy (k={args.k}): {test_acc:.2%}")
    print(f"Mean (vuln, fix) cosine similarity: {test_pair_sim:.4f}  "
          f"(lower = better separation)")

    corpus_path = Path("corpus_instr_focal_embeddings.npz")
    np.savez(corpus_path,
             embeddings=corpus_emb.numpy(),
             labels=corpus_lab.numpy())
    print(f"Corpus saved -> {corpus_path.resolve()}\n")

    print("--- Results ---")
    print(f"  §6  BigVul block-level Triplet k-NN:              51.21%  pair-sim 0.979->0.986")
    print(f"  §8  BigVul instr-level Triplet k-NN:              48.39%  pair-sim 0.9984->0.9995")
    print(f"  §10b BigVul instr-level FCL+SAG (k={args.k}):  {test_acc:.2%}  "
          f"pair-sim {test_pair_sim:.4f}")

    if test_pair_sim < 0.95:
        print("Pair-sim decreased significantly — FCL+SAGPooling breaks collapse.")
    elif test_pair_sim < 0.9984:
        print("Pair-sim improved over §8 baseline (0.9984) — partial improvement.")
    else:
        print("Pair-sim did not improve — dilution or data starvation remains binding.")

    if test_acc >= 0.62:
        print("FCL+SAGPooling closes the gap to CodeBERT — structural signal is real.")
    elif test_acc >= 0.58:
        print("Improvement over instruction-level classifier baseline — contrastive adds value.")
    elif test_acc >= 0.51:
        print("Improvement over §6 block-level triplet — instruction granularity helps.")
    else:
        print("No improvement over §8 — collapse persists; identifier features likely needed.")


if __name__ == "__main__":
    main()
