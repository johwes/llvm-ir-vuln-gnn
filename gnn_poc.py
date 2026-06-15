#!/usr/bin/env python3
"""
Minimal GNN proof-of-concept — pure numpy, no pip installs.

Pipeline:
  LLVM IR text  →  CFG adjacency matrix + node features
               →  1-layer graph convolution (GCN)
               →  global mean pooling  →  graph embedding
               →  binary classifier   →  vuln / fixed

With only 10 samples (5 pairs), we use leave-one-out cross-validation:
train on 9, predict the held-out sample, repeat 10 times.

Usage:
    python3 gnn_poc.py [ir_dir]
"""

import sys
import math
import random
import numpy as np
from pathlib import Path

# Reuse graph extraction from the companion script
sys.path.insert(0, str(Path(__file__).parent))
from graph_demo import extract_graphs

HIDDEN = 8
LR     = 0.05
EPOCHS = 300
SEED   = 42


# ---------------------------------------------------------------------------
# Graph → numpy
# ---------------------------------------------------------------------------

def build_graph(g: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (A_norm, X) for one extracted graph.

    A_norm : symmetrically normalised adjacency with self-loops (GCN-style)
    X      : node features — [out_degree, in_degree, 1] per basic block
    """
    blocks    = g["blocks"]
    n         = len(blocks)
    block_idx = {b: i for i, b in enumerate(blocks)}

    A = np.zeros((n, n))
    for src, dst in g["cfg_edges"]:
        si, di = block_idx.get(src, -1), block_idx.get(dst, -1)
        if si >= 0 and di >= 0:
            A[si, di] = 1.0
            A[di, si] = 1.0      # treat as undirected

    # Add self-loops then normalise: D^-0.5 (A+I) D^-0.5
    A_hat     = A + np.eye(n)
    deg       = A_hat.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)))
    A_norm    = D_inv_sqrt @ A_hat @ D_inv_sqrt

    # Node features: [out_degree, in_degree, constant]
    out_deg = A.sum(axis=1, keepdims=True)
    in_deg  = A.sum(axis=0, keepdims=True).T
    X       = np.hstack([out_deg, in_deg, np.ones((n, 1))])   # (n, 3)

    return A_norm, X


# ---------------------------------------------------------------------------
# Minimal GCN: 1 conv layer → mean pool → linear → sigmoid
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def forward(A_norm, X, W1, w2):
    H    = np.maximum(0, A_norm @ X @ W1)   # (n, HIDDEN)  relu
    h    = H.mean(axis=0)                    # (HIDDEN,)    mean pool
    prob = sigmoid(h @ w2)                   # scalar
    return H, h, prob

def backward(A_norm, X, W1, w2, H, h, prob, y):
    d_logit = prob - y                                # scalar
    dw2     = h * d_logit                             # (HIDDEN,)
    dh      = w2 * d_logit                            # (HIDDEN,)
    n       = H.shape[0]
    dH      = np.outer(np.ones(n) / n, dh)            # (n, HIDDEN)  mean pool grad
    dH[H <= 0] = 0.0                                  # relu grad
    dW1     = (A_norm @ X).T @ dH                     # (3, HIDDEN)
    return dW1, dw2

def bce(prob, y):
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ir_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ir")
    rng    = random.Random(SEED)
    np.random.seed(SEED)

    # Load all graphs
    dataset = []   # [(label, name, A_norm, X)]
    for ll in sorted(ir_dir.glob("*.ll")):
        for fname, g in extract_graphs(ll).items():
            label = 1 if "_vuln" in ll.stem else 0
            A_norm, X = build_graph(g)
            dataset.append((label, ll.stem, A_norm, X))

    if len(dataset) < 2:
        print("Need at least 2 .ll files.")
        sys.exit(1)

    print(f"Loaded {len(dataset)} graphs  "
          f"({sum(d[0] for d in dataset)} vuln, "
          f"{sum(1-d[0] for d in dataset)} fixed)\n")
    print(f"Model: GCN(3→{HIDDEN}) → mean-pool → linear → sigmoid")
    print(f"Train: leave-one-out, {EPOCHS} epochs, lr={LR}\n")

    # Leave-one-out cross-validation
    correct = 0
    rows    = []

    for held_out in range(len(dataset)):
        train = [d for i, d in enumerate(dataset) if i != held_out]

        # Fresh random weights for each fold
        W1 = np.random.randn(3, HIDDEN) * 0.1
        w2 = np.random.randn(HIDDEN)    * 0.1

        for _ in range(EPOCHS):
            rng.shuffle(train)
            for (y, _, A_norm, X) in train:
                H, h, prob = forward(A_norm, X, W1, w2)
                dW1, dw2   = backward(A_norm, X, W1, w2, H, h, prob, y)
                W1 -= LR * dW1
                w2 -= LR * dw2

        y_true, name, A_norm, X = dataset[held_out]
        _, _, prob = forward(A_norm, X, W1, w2)
        pred       = 1 if prob >= 0.5 else 0
        ok         = pred == y_true
        correct   += int(ok)
        rows.append((name, y_true, prob, pred, ok))

    # Results table
    print(f"{'SAMPLE':<24} {'TRUE':>6} {'PROB':>7} {'PRED':>6} {'':>4}")
    print("-" * 52)
    for name, y_true, prob, pred, ok in rows:
        label_str = "vuln " if y_true == 1 else "fixed"
        pred_str  = "vuln " if pred   == 1 else "fixed"
        mark      = "✓" if ok else "✗"
        print(f"  {name:<22} {label_str:>6} {prob:>7.3f} {pred_str:>6}  {mark}")

    acc = correct / len(dataset)
    print(f"\n  Leave-one-out accuracy: {correct}/{len(dataset)} = {acc:.0%}")
    print()

    if acc >= 0.7:
        print("  ✓ GNN can separate vuln from fixed on this graph structure.")
    else:
        print("  — Accuracy low; expected for cases where fixes don't add branches")
        print("    (doublefree, uninit). Those are invisible at the CFG level.")


if __name__ == "__main__":
    main()
