#!/usr/bin/env python3
"""
train_contrastive.py — Supervised Contrastive Learning on Devign IR graphs.

Instead of f(graph) → {0,1}, trains f(graph) → embedding_vector where
geometry encodes vulnerability: same-class graphs cluster, cross-class graphs
are pushed apart. At inference, new functions are classified by k-nearest-
neighbour against a reference corpus — no decision boundary, no threshold.

Why this fits SCAR: each accepted patch produces a (vuln_IR, fix_IR) pair.
Embedding the vulnerable version and inserting it into the corpus takes
milliseconds. Coverage grows continuously without touching model weights.

Architecture:
  Encoder:  RGCNConv(45→64) → RGCNConv(64→64) → AttentionalAggregation → 64-dim
  Proj head (training only): Linear(64→128) → ReLU → Linear(128→128) → L2-norm
  Inference: encoder output L2-normalized, cosine k-NN against corpus

Loss: Supervised Contrastive (Khosla et al. 2020)
  Pulls same-label pairs together, pushes cross-label pairs apart.
  Requires no explicit (vuln, fix) commit pairing — uses labels within batch.

Evaluation: k-NN accuracy against training corpus (k=5).
  After training: saves encoder weights + corpus embeddings as numpy arrays
  ready for FAISS / Qdrant ingestion.

Usage:
    python train_contrastive.py                 # full dataset, 50 epochs
    python train_contrastive.py --epochs 20     # quick run
    python train_contrastive.py --batch-size 64 # larger batches = more pos pairs
    python train_contrastive.py --temp 0.1      # higher temperature = softer loss
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv
from torch_geometric.nn.aggr import AttentionalAggregation

HERE = Path(__file__).parent
DATA = HERE / "data"

N_FEATURES = 45  # must match preprocess.py block-level graphs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ContrastiveGNN(nn.Module):
    """PDG encoder + projection head for supervised contrastive training.

    encode() returns the 64-dim graph embedding (used at inference).
    forward() returns the L2-normalised projected embedding (used for loss).
    The projection head is discarded after training — standard SimCLR/SupCon
    practice: the encoder generalises better without the linear projection.
    """

    def __init__(self, in_features: int = N_FEATURES,
                 hidden: int = 64, proj_dim: int = 128):
        super().__init__()
        # Encoder — same architecture as train.py
        self.conv1 = RGCNConv(in_features, hidden, num_relations=2)
        self.conv2 = RGCNConv(hidden, hidden, num_relations=2)
        gate_nn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )
        self.pool = AttentionalAggregation(gate_nn=gate_nn)

        # Projection head — used only during training
        self.proj = nn.Sequential(
            nn.Linear(hidden, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def encode(self, x, edge_index, edge_type, batch) -> torch.Tensor:
        """Graph → 64-dim embedding. Use this at inference time."""
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        return self.pool(x, batch)                        # (B, hidden)

    def forward(self, x, edge_index, edge_type, batch) -> torch.Tensor:
        """Graph → L2-normalised projected embedding. Used for SupCon loss."""
        h = self.encode(x, edge_index, edge_type, batch)
        z = self.proj(h)
        return F.normalize(z, dim=-1)                     # (B, proj_dim)


# ---------------------------------------------------------------------------
# Supervised Contrastive loss
# ---------------------------------------------------------------------------

def supcon_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.07) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al. 2020, NeurIPS).

    embeddings : (N, D) L2-normalised
    labels     : (N,)  integer class labels {0, 1}
    temperature: τ — lower = harder separation (0.07 is standard)

    For each anchor i, positives = same-label samples (excluding self).
    Loss = mean over anchors of: -1/|P(i)| * sum_p log [exp(sim(i,p)/τ) /
                                                          sum_{a≠i} exp(sim(i,a)/τ)]
    Anchors with no positive (only label in batch) are excluded from the mean.
    """
    N = embeddings.shape[0]
    device = embeddings.device

    sim = torch.mm(embeddings, embeddings.T) / temperature     # (N, N)

    # Masks
    eye     = torch.eye(N, dtype=torch.bool, device=device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    neg_mask = ~eye

    # Numerical stability: subtract row-wise max before exp
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim  = torch.exp(sim) * neg_mask                       # zero out self
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    n_pos = pos_mask.sum(dim=1).float()
    loss  = -(log_prob * pos_mask).sum(dim=1) / (n_pos + 1e-8)

    # Only average over anchors that actually have a positive pair in this batch
    valid = n_pos > 0
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    return loss[valid].mean()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graphs(pkl_path: Path) -> list[Data]:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    dataset = []
    for g in raw:
        x          = torch.tensor(g["x"],         dtype=torch.float)
        edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
        edge_type  = torch.tensor(g["edge_type"],  dtype=torch.long)
        y          = torch.tensor([g["y"]],        dtype=torch.float)

        if x.shape[0] > 1:
            x = (x - x.mean(0)) / (x.std(0) + 1e-8)

        dataset.append(Data(x=x, edge_index=edge_index,
                            edge_type=edge_type, y=y))
    return dataset


# ---------------------------------------------------------------------------
# k-NN evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_corpus(model: ContrastiveGNN, loader: DataLoader,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed all graphs in loader → (embeddings, labels) on CPU."""
    model.eval()
    embs, labs = [], []
    for batch in loader:
        batch = batch.to(device)
        h = model.encode(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        embs.append(h.cpu())
        labs.append(batch.y.squeeze().cpu())
    embs = F.normalize(torch.cat(embs, dim=0), dim=-1)
    labs = torch.cat(labs, dim=0)
    return embs, labs


@torch.no_grad()
def knn_accuracy(model: ContrastiveGNN, corpus_emb: torch.Tensor,
                 corpus_lab: torch.Tensor, eval_loader: DataLoader,
                 device: torch.device, k: int = 5) -> float:
    """k-NN accuracy: majority vote among k nearest corpus neighbours."""
    model.eval()
    correct = total = 0
    for batch in eval_loader:
        batch = batch.to(device)
        h = model.encode(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        h = F.normalize(h, dim=-1).cpu()

        sims    = torch.mm(h, corpus_emb.T)              # (batch, N_corpus)
        _, topk = sims.topk(k, dim=1)
        knn_lab = corpus_lab[topk].float()               # (batch, k)
        preds   = (knn_lab.mean(dim=1) >= 0.5).long()
        labels  = batch.y.squeeze().long().cpu()

        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(model: ContrastiveGNN, loader: DataLoader,
                optimizer, device: torch.device,
                temperature: float) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        z      = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        labels = batch.y.squeeze().long()
        loss   = supcon_loss(z, labels, temperature)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=50)
    ap.add_argument("--hidden",     type=int,   default=64)
    ap.add_argument("--proj-dim",   type=int,   default=128)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int,   default=64,
                    help="Larger = more positive pairs per batch (recommended ≥32)")
    ap.add_argument("--temp",       type=float, default=0.07,
                    help="SupCon temperature τ (0.07 = standard, higher = softer)")
    ap.add_argument("--k",          type=int,   default=5,
                    help="k for k-NN evaluation")
    ap.add_argument("--checkpoint", type=str,   default="encoder_contrastive.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    for split in ["train", "valid", "test"]:
        if not (DATA / f"{split}_graphs.pkl").exists():
            print(f"Missing data/{split}_graphs.pkl — run preprocess.py first.")
            sys.exit(1)

    print("Loading graphs ...")
    train_data = load_graphs(DATA / "train_graphs.pkl")
    valid_data = load_graphs(DATA / "valid_graphs.pkl")
    test_data  = load_graphs(DATA / "test_graphs.pkl")
    print(f"  train={len(train_data)}  valid={len(valid_data)}  test={len(test_data)}")

    vuln  = sum(1 for d in train_data if d.y.item() == 1)
    fixed = len(train_data) - vuln
    print(f"  train class balance: {vuln} vuln / {fixed} fixed\n")

    # Larger batch = more positive pairs per batch = better SupCon signal
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    # Corpus loader uses full batch for speed (no shuffle needed)
    corpus_loader = DataLoader(train_data, batch_size=256, shuffle=False)
    valid_loader  = DataLoader(valid_data, batch_size=256)
    test_loader   = DataLoader(test_data,  batch_size=256)

    model     = ContrastiveGNN(N_FEATURES, args.hidden, args.proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ContrastiveGNN(in={N_FEATURES}, hidden={args.hidden}, "
          f"proj={args.proj_dim})  params={n_params:,}")
    print(f"Loss:  SupCon  τ={args.temp}  batch={args.batch_size}  k={args.k}\n")

    best_val_acc = 0.0
    checkpoint   = Path(args.checkpoint)

    print(f"{'Epoch':>5}  {'Loss':>8}  {'Val k-NN':>9}  {'':>6}")
    print("-" * 38)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, args.temp)
        scheduler.step()

        # k-NN validation: build corpus from training embeddings each epoch
        corpus_emb, corpus_lab = build_corpus(model, corpus_loader, device)
        val_acc = knn_accuracy(model, corpus_emb, corpus_lab,
                               valid_loader, device, args.k)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)
            marker = "← best"

        print(f"{epoch:>5}  {loss:>8.4f}  {val_acc:>9.2%}  {marker}")

    # Final test evaluation
    print(f"\nLoading best checkpoint ({checkpoint}) ...")
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    corpus_emb, corpus_lab = build_corpus(model, corpus_loader, device)
    test_acc = knn_accuracy(model, corpus_emb, corpus_lab,
                            test_loader, device, args.k)
    print(f"Test k-NN accuracy (k={args.k}): {test_acc:.2%}")

    # Save corpus embeddings for FAISS / Qdrant ingestion
    corpus_path = Path("corpus_embeddings.npz")
    np.savez(corpus_path,
             embeddings=corpus_emb.numpy(),
             labels=corpus_lab.numpy())
    print(f"Corpus embeddings saved → {corpus_path.resolve()}")
    print(f"  shape: {corpus_emb.shape}  (ready for FAISS IndexFlatIP)\n")

    print("--- Comparison ---")
    print(f"  Classifier GNN best (4d, BCE):               57.84%")
    print(f"  CodeBERT (4a, fine-tuned):                   63.43%")
    print(f"  Contrastive GNN (this run, k-NN k={args.k}):  {test_acc:.2%}")

    if test_acc >= 0.60:
        print("✓ Contrastive objective improves over classifier — embedding "
              "space generalises better than a BCE decision boundary")
    elif test_acc >= 0.58:
        print("— Marginal improvement; embedding space partially separates classes")
    else:
        print("— Below classifier baseline; the contrastive loss may need "
              "larger batches, lower temperature, or more epochs")

    print(f"\nNext steps:")
    print(f"  faiss.IndexFlatIP — load corpus_embeddings.npz, query new IR graphs")
    print(f"  Each accepted SCAR patch: embed vuln IR → append to corpus (no retraining)")


if __name__ == "__main__":
    main()
