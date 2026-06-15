#!/usr/bin/env python3
"""
Experiment B — Instruction-level node density check.

Uses the existing pkl files (block-level) to measure current graph sizes
and estimate what instruction-level graphs will look like.

Run from the train_gnn/ directory after preprocess.py has been run.
No llvmlite or clang required.

Usage:
    python exp_b_density.py
"""

import pickle
import statistics
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"


def report(label, counts):
    s = sorted(counts)
    n = len(s)
    print(f"\n{label} ({n} functions):")
    print(f"  min:    {s[0]}")
    print(f"  median: {statistics.median(s):.0f}")
    print(f"  p75:    {s[int(n * 0.75)]}")
    print(f"  p90:    {s[int(n * 0.90)]}")
    print(f"  p95:    {s[int(n * 0.95)]}")
    print(f"  max:    {s[-1]}")


def main():
    for split in ["train", "valid", "test"]:
        pkl = DATA / f"{split}_graphs.pkl"
        if not pkl.exists():
            print(f"Missing {pkl} — run preprocess.py first.")
            return

    # Block-level node counts from existing pkl
    all_block_counts = []
    for split in ["train", "valid", "test"]:
        with open(DATA / f"{split}_graphs.pkl", "rb") as f:
            graphs = pickle.load(f)
        counts = [len(g["x"]) for g in graphs]
        all_block_counts.extend(counts)
        print(f"{split}: {len(graphs)} functions, "
              f"median blocks={statistics.median(counts):.0f}, "
              f"max={max(counts)}")

    report("Block-level nodes (current)", all_block_counts)

    # Instruction-level estimate: empirically ~7 instructions per block
    # for Devign -O0 IR (mix of alloca, load, icmp, call, store, br).
    # Multiply by 7 to get a conservative upper bound.
    est_instr = [n * 7 for n in all_block_counts]
    report("Estimated instruction-level nodes (×7)", est_instr)

    s = sorted(all_block_counts)
    n = len(s)
    large = sum(1 for c in all_block_counts if c > 100)
    print(f"\n  Functions with >100 blocks (>~700 instr nodes): "
          f"{large}/{n} ({100*large/n:.1f}%)")
    large2 = sum(1 for c in all_block_counts if c > 200)
    print(f"  Functions with >200 blocks (>~1400 instr nodes): "
          f"{large2}/{n} ({100*large2/n:.1f}%)")

    print("\nBatch size guidance for instruction-level training:")
    p90 = sorted(est_instr)[int(n * 0.90)]
    print(f"  p90 node count ≈ {p90}")
    if p90 < 500:
        print("  → batch_size=32 should be safe on GPU; 8-16 on CPU")
    elif p90 < 1500:
        print("  → batch_size=16 on GPU; 4-8 on CPU (watch VRAM)")
    else:
        print("  → batch_size=4-8 on GPU; may need graph truncation")


if __name__ == "__main__":
    main()
