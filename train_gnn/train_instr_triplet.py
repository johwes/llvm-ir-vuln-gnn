#!/usr/bin/env python3
"""
train_instr_triplet.py — Triplet Contrastive Learning on BigVul instruction-level graphs.

§8 experiment: does triplet loss succeed at instruction granularity where it failed at
block level? At block level, (vuln, fix) pair cosine similarity was 0.979→0.986 (wrong
direction) because a 3-line patch leaves the block adjacency matrix near-invariant.
At instruction level, the same patch changes 3-5 nodes in a ~300-node graph, so pairs
are now structurally distinguishable.

Architecture: InstructionContrastiveGNN
  Embedding:  nn.Embedding(110, 128) <- opcode vocab (no float features)
  Encoder:    RGCNConv(128->64, 3 rels) x 2 + AttentionalAggregation
  Proj head:  Linear(64->128) -> ReLU -> Linear(128->128) -> L2-norm  (training only)
  Inference:  encoder output, cosine k-NN

Loss: Triplet (same design as train_triplet.py)
  anchor   = vulnerable function instruction graph
  negative = its exact patched counterpart (same commit)
  positive = another vuln function, same CWE preferred

Dataset: BigVul instruction-level pairs from preprocess_instr_bigvul.py
  data/bigvul_{train,valid,test}_instr_pairs.pkl

Usage:
    python train_instr_triplet.py                          # 50 epochs, defaults
    python train_instr_triplet.py --epochs 30 --hidden 64  # faster first run
    python train_instr_triplet.py --batch-size 128
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
from torch_geometric.nn import RGCNConv
from torch_geometric.nn.aggr import AttentionalAggregation

HERE = Path(__file__).parent
DATA = HERE / "data"

VOCAB   = 110
PAD_IDX = 79


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class InstructionContrastiveGNN(nn.Module):
    """
    Instruction-level GNN encoder + projection head for triplet training.

    encode() returns the hidden-dim graph embedding (used at inference).
    forward() returns the L2-normalised projected embedding (used for loss).
    """

    def __init__(self, vocab: int = VOCAB, embed_dim: int = 128,
                 hidden: int = 64, proj_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=PAD_IDX)
        self.conv1 = RGCNConv(embed_dim, hidden, num_relations=3)
        self.conv2 = RGCNConv(hidden,    hidden, num_relations=3)
        gate_nn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )
        self.pool = AttentionalAggregation(gate_nn=gate_nn)
        self.proj = nn.Sequential(
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
        return self.pool(h, batch)                             # (B, hidden)

    def forward(self, x, edge_index, edge_type, batch) -> torch.Tensor:
        """Returns L2-normalised projected embedding. Used for triplet loss."""
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
        x          = torch.tensor(g["x"],         dtype=torch.long)   # opcode indices
        edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
        edge_type  = torch.tensor(g["edge_type"],  dtype=torch.long)
        # No z-score: x is integer vocab indices, not floats
        return Data(x=x, edge_index=edge_index, edge_type=edge_type)


def pair_collate(batch):
    """Collate (vuln, fix, cwe) triples into two PyG batches + cwe list."""
    vulns, fixes, cwes = zip(*batch)
    return Batch.from_data_list(list(vulns)), Batch.from_data_list(list(fixes)), list(cwes)


def load_pairs(pkl_path: Path) -> list:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Triplet loss
# ---------------------------------------------------------------------------

def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor,
                 negative: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """
    Cosine-similarity triplet loss on L2-normalised embeddings.

    anchor   - vuln embedding
    positive - same-CWE vuln embedding (or random vuln)
    negative - exact fix embedding for the anchor

    Loss = mean over active triplets of: max(0, sim(a,neg) - sim(a,pos) + margin)
    """
    pos_sim = (anchor * positive).sum(dim=-1)
    neg_sim = (anchor * negative).sum(dim=-1)
    loss    = torch.clamp(neg_sim - pos_sim + margin, min=0.0)
    active  = loss > 0
    if not active.any():
        return loss.mean()
    return loss[active].mean()


# ---------------------------------------------------------------------------
# Positive mining: same-CWE vuln in batch, fall back to nearest other
# ---------------------------------------------------------------------------

def mine_positives(vuln_emb: torch.Tensor, cwes: list) -> torch.Tensor:
    """
    For each anchor i, find another vuln j with the same CWE.
    Fall back to the nearest other vuln (by cosine sim) if no same-CWE exists.
    Returns a (N, D) tensor of positive embeddings.
    """
    N = vuln_emb.shape[0]
    positives = torch.zeros_like(vuln_emb)

    cwe_to_idx = {}
    for i, cwe in enumerate(cwes):
        cwe_to_idx.setdefault(cwe, []).append(i)

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

def train_epoch(model: InstructionContrastiveGNN, loader,
                optimizer, device: torch.device, margin: float) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for vuln_batch, fix_batch, cwes in loader:
        vuln_batch = vuln_batch.to(device)
        fix_batch  = fix_batch.to(device)
        optimizer.zero_grad()

        vuln_emb = model(vuln_batch.x, vuln_batch.edge_index,
                         vuln_batch.edge_type, vuln_batch.batch)
        fix_emb  = model(fix_batch.x,  fix_batch.edge_index,
                         fix_batch.edge_type,  fix_batch.batch)

        positive_emb = mine_positives(vuln_emb.detach(), cwes).to(device)

        loss = triplet_loss(vuln_emb, positive_emb, fix_emb, margin)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def pair_similarity(model: InstructionContrastiveGNN, loader,
                    device: torch.device) -> float:
    """Mean cosine similarity between (vuln, fix) pairs -- should decrease."""
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
    """
    Build a corpus of (vuln=1, fix=0) embeddings from the training loader.
    Returns (embeddings, labels) both L2-normalised.
    """
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

    embs = torch.cat([vuln_embs, fix_embs], dim=0)
    labs = torch.cat([
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
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int,   default=64)
    ap.add_argument("--margin",     type=float, default=0.3,
                    help="Triplet margin (cosine scale 0-2)")
    ap.add_argument("--k",          type=int,   default=5)
    ap.add_argument("--checkpoint", type=str,   default="encoder_instr_triplet.pt")
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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: InstructionContrastiveGNN(vocab={VOCAB}, embed={args.embed_dim}, "
          f"hidden={args.hidden}, proj={args.proj_dim})  params={n_params:,}")
    print(f"Loss:  Triplet  margin={args.margin}  batch={args.batch_size}  k={args.k}\n")

    best_val_acc = 0.0
    checkpoint   = Path(args.checkpoint)

    print(f"{'Epoch':>5}  {'Loss':>8}  {'Pair-Sim':>9}  {'Val k-NN':>9}  {'':>6}")
    print("-" * 48)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, args.margin)
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
    test_acc = knn_accuracy(model, corpus_emb, corpus_lab,
                            test_loader, device, args.k)
    test_pair_sim = pair_similarity(model, test_loader, device)

    print(f"\nTest k-NN accuracy (k={args.k}): {test_acc:.2%}")
    print(f"Mean (vuln, fix) cosine similarity: {test_pair_sim:.4f}  "
          f"(lower = better separation)")

    corpus_path = Path("corpus_instr_triplet_embeddings.npz")
    np.savez(corpus_path,
             embeddings=corpus_emb.numpy(),
             labels=corpus_lab.numpy())
    print(f"Corpus saved -> {corpus_path.resolve()}\n")

    print("--- Results ---")
    print(f"  BigVul block-level Triplet k-NN (section 6):     51.21%  pair-sim 0.979->0.986")
    print(f"  Instruction-level GNN classifier (section 7):    58.00%")
    print(f"  BigVul instr-level Triplet k-NN (section 8, k={args.k}):  {test_acc:.2%}  "
          f"pair-sim {test_pair_sim:.4f}")

    if test_pair_sim < 0.95:
        print("Pair-sim decreased significantly -- triplet loss is separating vuln/fix "
              "at instruction granularity (hypothesis confirmed)")
    elif test_pair_sim < 0.979:
        print("Pair-sim improved over block-level baseline (0.979) -- "
              "instruction-level representation provides more signal")
    else:
        print("Pair-sim did not improve -- structural diff still below resolution "
              "or training collapsed")

    if test_acc >= 0.62:
        print("Instruction-level triplet closes the gap to CodeBERT -- "
              "structural diff at instruction granularity is a strong signal")
    elif test_acc >= 0.58:
        print("Improvement over instruction-level classifier baseline -- "
              "contrastive objective adds value at this granularity")
    elif test_acc >= 0.51:
        print("Improvement over block-level triplet baseline -- "
              "instruction granularity helps but ceiling remains")
    else:
        print("No improvement over block-level -- collapse or insufficient data")


if __name__ == "__main__":
    main()
