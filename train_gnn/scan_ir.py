#!/usr/bin/env python3
"""
scan_ir.py — Run the trained GNN on a compiled LLVM IR file.

Usage:
    python scan_ir.py function.ll
    python scan_ir.py function.ll --threshold 0.4
    python scan_ir.py function.ll --model path/to/model.pt
    python scan_ir.py file.ll --all-functions          # score every function, ranked
"""

import argparse
import re
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from preprocess import ir_to_graph
from train import DefectGNN, N_FEATURES

_FN_NAME_RE = re.compile(r'@([\w.]+)\s*\(')


def _score_graph(g, model):
    x          = torch.tensor(g["x"],         dtype=torch.float)
    edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
    edge_type  = torch.tensor(g["edge_type"],  dtype=torch.long)
    if x.shape[0] > 1:
        x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    batch = torch.zeros(x.shape[0], dtype=torch.long)
    with torch.no_grad():
        prob = torch.sigmoid(model(x, edge_index, edge_type, batch)).item()
    return x.shape[0], prob


def scan_all_fns(ir_text, model, threshold):
    """Score every non-declaration function in the IR. Returns list sorted by score desc."""
    results = []
    segs = re.split(r'(?=^define\b)', ir_text, flags=re.MULTILINE)
    for seg in segs:
        seg = seg.strip()
        if not seg.startswith("define"):
            continue
        m = _FN_NAME_RE.search(seg[:300])
        if not m:
            continue
        fn_name = m.group(1)
        g = ir_to_graph(seg)
        if g is None:
            continue
        n_blocks, prob = _score_graph(g, model)
        results.append((fn_name, n_blocks, prob))
    results.sort(key=lambda r: r[2], reverse=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ir_file", help=".ll file to scan")
    ap.add_argument("--model",         default=str(HERE / "model.pt"))
    ap.add_argument("--threshold",     type=float, default=0.5)
    ap.add_argument("--all-functions", action="store_true",
                    help="Score every function in the IR, sorted by score desc")
    args = ap.parse_args()

    ir_text = Path(args.ir_file).read_text(errors="replace")

    ckpt   = torch.load(args.model, map_location="cpu", weights_only=True)
    hidden = ckpt["lin.weight"].shape[1]
    model  = DefectGNN(N_FEATURES, hidden=hidden)
    model.load_state_dict(ckpt)
    model.eval()

    if args.all_functions:
        results = scan_all_fns(ir_text, model, args.threshold)
        if not results:
            print("ERROR: no scoreable functions found in IR")
            sys.exit(1)
        for fn_name, n_blocks, prob in results:
            label = "VULNERABLE" if prob >= args.threshold else "safe"
            print(f"  {fn_name:<40s} [{n_blocks:>3} blocks]  {prob:.1%}  ->  {label}")
        sys.exit(0)

    g = ir_to_graph(ir_text)
    if g is None:
        print("ERROR: could not parse IR into a graph (no basic blocks found)")
        sys.exit(1)

    n_blocks, prob = _score_graph(g, model)
    label = "VULNERABLE" if prob >= args.threshold else "safe"
    print(f"{Path(args.ir_file).name}  [{n_blocks} blocks]  {prob:.1%}  \u2192  {label}")


if __name__ == "__main__":
    main()
