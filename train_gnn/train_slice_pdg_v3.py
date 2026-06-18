#!/usr/bin/env python3
"""
train_slice_pdg_v3.py — §23 PDG slice GNN with sink-node readout.

Implements the two architectural changes recommended by the external reviewer
on top of §12 (train_slice_pdg.py):

1. Sink-node readout (replaces global AttentionalAggregation).
   preprocess_slice_pdg_v3.py tags which nodes in each slice are the identified
   dangerous sinks.  After RGCN message passing, the embedding of each sink
   node already aggregates its K-hop neighborhood (data-flow origins + local
   guard conditions).  Instead of pooling the entire slice, we apply the
   classifier only to sink node embeddings and scatter-max over sinks per graph.
   This eliminates signal dilution from the 3,105-node worst-case slices in §12.

   For fallback graphs (no sinks found), all nodes are marked as sinks so
   scatter-max degenerates to global max pool — a conservative default.

2. Residual connections + LayerNorm on both RGCN layers.
   Standard remedy for oversmoothing in medium-to-large graphs: LayerNorm
   stabilises activations and the residual connection preserves the original
   opcode semantics in the node embedding while safely accumulating relational
   neighborhood context.

Architecture:
  Embedding(110, embed_dim) → Linear(embed_dim, hidden) [proj]
  RGCNConv(hidden, hidden, 3) + LayerNorm + residual  [layer 1]
  RGCNConv(hidden, hidden, 3) + LayerNorm + residual  [layer 2]
  Linear(hidden, 1) applied to sink nodes only
  scatter_max over sink nodes per graph → logit per graph

Usage:
    python preprocess_slice_pdg_v3.py --skip-download   # first
    python train_slice_pdg_v3.py --epochs 30 --hidden 64
    python train_slice_pdg_v3.py --epochs 60 --hidden 128
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_max_pool

HERE = Path(__file__).parent
DATA = HERE / "data"

VOCAB_SIZE = 110


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SlicePDGGNNv3(nn.Module):
    """
    PDG slice GNN with sink-node readout and residual + LayerNorm.

    Key change over §12 SlicePDGGNN:
    - No AttentionalAggregation over the whole graph.
    - Logit computed only from sink node embeddings; scatter-max per graph.
    """

    def __init__(self, vocab: int = VOCAB_SIZE, embed_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=79)
        self.proj  = nn.Linear(embed_dim, hidden, bias=False)
        self.conv1 = RGCNConv(hidden, hidden, num_relations=3)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = RGCNConv(hidden, hidden, num_relations=3)
        self.norm2 = nn.LayerNorm(hidden)
        self.lin   = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_type, batch, sink_mask):
        h = F.relu(self.proj(self.embed(x.squeeze(-1))))   # (N, hidden)

        # Layer 1: RGCN + LayerNorm + residual
        h1 = self.norm1(F.relu(self.conv1(h, edge_index, edge_type)))
        h  = h + F.dropout(h1, p=0.3, training=self.training)

        # Layer 2: RGCN + LayerNorm + residual
        h2 = self.norm2(F.relu(self.conv2(h, edge_index, edge_type)))
        h  = h + F.dropout(h2, p=0.3, training=self.training)

        # Sink-node readout: classify sinks only, scatter-max per graph
        sink_h    = h[sink_mask]                              # (S, hidden)
        sink_gid  = batch[sink_mask]                          # (S,)
        per_sink  = self.lin(sink_h)                          # (S, 1)
        num_graphs = int(batch.max().item()) + 1
        # global_max_pool: max over nodes per graph (here: sink nodes only)
        logits = global_max_pool(per_sink, sink_gid, size=num_graphs).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graphs(pkl_path: Path) -> list[Data]:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    dataset = []
    for g in raw:
        x          = torch.tensor(g["x"],          dtype=torch.long)
        edge_index = torch.tensor(g["edge_index"],  dtype=torch.long)
        edge_type  = torch.tensor(g["edge_type"],   dtype=torch.long)
        y          = torch.tensor([g["y"]],         dtype=torch.float)
        sink_mask  = torch.tensor(g["sink_mask"],   dtype=torch.bool)
        dataset.append(Data(x=x, edge_index=edge_index,
                            edge_type=edge_type, y=y, sink_mask=sink_mask))
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
        logits = model(batch.x, batch.edge_index, batch.edge_type,
                       batch.batch, batch.sink_mask)
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
        logits = model(batch.x, batch.edge_index, batch.edge_type,
                       batch.batch, batch.sink_mask)
        preds  = (logits > 0).long()
        labels = batch.y.squeeze().long()
        correct += (preds == labels).sum().item()
        total   += batch.num_graphs
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=30)
    ap.add_argument("--hidden",     type=int,   default=64)
    ap.add_argument("--embed-dim",  type=int,   default=128)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int,   default=32)
    ap.add_argument("--checkpoint", type=str,   default="model_slice_pdg_v3.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    for split in ["train", "valid", "test"]:
        if not (DATA / f"{split}_slice_pdg_v3_graphs.pkl").exists():
            print(f"Missing data/{split}_slice_pdg_v3_graphs.pkl "
                  f"-- run preprocess_slice_pdg_v3.py first.")
            sys.exit(1)

    print("Loading PDG v3 slice graphs ...")
    train_data = load_graphs(DATA / "train_slice_pdg_v3_graphs.pkl")
    valid_data = load_graphs(DATA / "valid_slice_pdg_v3_graphs.pkl")
    test_data  = load_graphs(DATA / "test_slice_pdg_v3_graphs.pkl")
    print(f"  train={len(train_data)}  valid={len(valid_data)}  test={len(test_data)}")

    node_counts = [d.x.shape[0] for d in train_data]
    sink_counts = [d.sink_mask.sum().item() for d in train_data]
    import statistics
    print(f"  Slice sizes:  mean={statistics.mean(node_counts):.0f}  "
          f"median={statistics.median(node_counts):.0f}  max={max(node_counts)}")
    print(f"  Sink counts:  mean={statistics.mean(sink_counts):.1f}  "
          f"median={statistics.median(sink_counts):.0f}  max={max(sink_counts)}")

    vuln_train  = sum(1 for d in train_data if d.y.item() == 1)
    fixed_train = len(train_data) - vuln_train
    pos_weight  = torch.tensor([fixed_train / vuln_train]).to(device)
    print(f"  train class balance: {vuln_train} vuln / {fixed_train} fixed")
    print(f"  pos_weight: {pos_weight.item():.3f}\n")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size)

    model     = SlicePDGGNNv3(VOCAB_SIZE, args.embed_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SlicePDGGNNv3(vocab={VOCAB_SIZE}, embed={args.embed_dim}, "
          f"hidden={args.hidden}, relations=3)  params={n_params:,}\n")

    best_val_acc = 0.0
    checkpoint   = Path(args.checkpoint)

    print(f"{'Epoch':>5}  {'Loss':>8}  {'Val Acc':>8}  {'':>6}")
    print("-" * 35)

    for epoch in range(1, args.epochs + 1):
        loss    = train_epoch(model, train_loader, optimizer, device, pos_weight)
        val_acc = evaluate(model, valid_loader, device)
        scheduler.step()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)
            marker = "<- best"

        print(f"{epoch:>5}  {loss:>8.4f}  {val_acc:>8.2%}  {marker}")

    print(f"\nLoading best checkpoint ({checkpoint}) ...")
    model.load_state_dict(torch.load(checkpoint, map_location=device,
                                     weights_only=True))
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.2%}")
    print(f"\nCheckpoint saved to: {checkpoint.resolve()}\n")

    print("--- Results ---")
    print(f"  §12 PDG slice (global AttentionalAggregation, 30ep h=64):   56.48%")
    print(f"  §23 PDG v3    (sink-node readout + residual/LN, this run):   {test_acc:.2%}")
    print()
    print("Run eval_all_models.py --scarnet to compare real-world scarnet performance.")
    print("Key metrics: P@13 on scarnet, rank of deflate_stored on zlib.")


if __name__ == "__main__":
    main()
