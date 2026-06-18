#!/usr/bin/env python3
"""
scan_ir.py — Run the trained GNN on a compiled LLVM IR file.

Usage:
    python scan_ir.py function.ll
    python scan_ir.py function.ll --threshold 0.4
    python scan_ir.py function.ll --model path/to/model.pt
    python scan_ir.py file.ll --all-functions          # score every function, ranked
    python scan_ir.py file.ll --context                # include slice vulnerability context
    python scan_ir.py file.ll --all-functions --context
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


# ---------------------------------------------------------------------------
# Model loading — auto-detect checkpoint type by key names
# ---------------------------------------------------------------------------

def _load_model(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    if "embed.weight" in ckpt:
        # SlicePDGGNN family (§12+)
        from train_slice_pdg import SlicePDGGNN, VOCAB_SIZE
        hidden = ckpt["lin.weight"].shape[1]
        embed_dim = ckpt["embed.weight"].shape[1]
        model = SlicePDGGNN(vocab=VOCAB_SIZE, embed_dim=embed_dim, hidden=hidden)
        model.load_state_dict(ckpt)
        model.eval()
        return model, "slice_pdg"
    else:
        hidden = ckpt["lin.weight"].shape[1]
        model = DefectGNN(N_FEATURES, hidden=hidden)
        model.load_state_dict(ckpt)
        model.eval()
        return model, "basic"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_graph(g, model, model_type: str):
    if model_type == "slice_pdg":
        x          = torch.tensor(g["x"],          dtype=torch.long)
        edge_index = torch.tensor(g["edge_index"],  dtype=torch.long)
        edge_type  = torch.tensor(g["edge_type"],   dtype=torch.long)
        batch      = torch.zeros(x.shape[0], dtype=torch.long)
        with torch.no_grad():
            prob = torch.sigmoid(model(x, edge_index, edge_type, batch)).item()
        return x.shape[0], prob
    else:
        x          = torch.tensor(g["x"],         dtype=torch.float)
        edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
        edge_type  = torch.tensor(g["edge_type"],  dtype=torch.long)
        if x.shape[0] > 1:
            x = (x - x.mean(0)) / (x.std(0) + 1e-8)
        batch = torch.zeros(x.shape[0], dtype=torch.long)
        with torch.no_grad():
            prob = torch.sigmoid(model(x, edge_index, edge_type, batch)).item()
        return x.shape[0], prob


def _context_for_ir(ir_text: str, fn_name: str) -> str | None:
    """Return formatted vulnerability context block for a named IR function, or None.

    Passes the full IR module so that cross-function calls (in multi-function
    files) remain visible to llvmlite and don't cause 'undefined value' errors.
    """
    try:
        from preprocess_slice_pdg import ir_to_graph_slice_pdg
        from slice_context import summarize_slice, format_for_llm
    except ImportError:
        return None

    g = ir_to_graph_slice_pdg(ir_text, fn_name=fn_name)
    if g is None:
        return None

    summary = summarize_slice(g, fn_name=fn_name)
    return format_for_llm(summary)


def scan_all_fns(ir_text, model, model_type, threshold, show_context):
    """Score every non-declaration function in the IR. Returns list sorted by score desc."""
    import llvmlite.binding as _llvm
    from preprocess_slice_pdg import ir_to_graph_slice_pdg

    # Parse full module once to enumerate functions — avoids per-segment splits
    # that lose cross-function declares present in multi-function IR files.
    try:
        _mod = _llvm.parse_assembly(ir_text)
    except Exception:
        return []
    fn_names = [fn.name for fn in _mod.functions if not fn.is_declaration]

    results = []
    for fn_name in fn_names:
        if model_type == "slice_pdg":
            g = ir_to_graph_slice_pdg(ir_text, fn_name=fn_name)
        else:
            # basic model: still needs per-function segment (different preprocessor)
            segs = re.split(r'(?=^define\b)', ir_text, flags=re.MULTILINE)
            g = None
            for seg in segs:
                m = _FN_NAME_RE.search(seg[:300])
                if m and m.group(1) == fn_name:
                    g = ir_to_graph(seg)
                    break
        if g is None:
            continue

        n_nodes, prob = _score_graph(g, model, model_type)

        ctx = None
        if show_context:
            from slice_context import summarize_slice, format_for_llm
            cg = ir_to_graph_slice_pdg(ir_text, fn_name=fn_name) if model_type != "slice_pdg" else g
            if cg is not None:
                ctx = format_for_llm(summarize_slice(cg, fn_name=fn_name), score=prob)

        results.append((fn_name, n_nodes, prob, ctx))
    results.sort(key=lambda r: r[2], reverse=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ir_file", help=".ll file to scan")
    ap.add_argument("--model",         default=str(HERE / "model_slice_pdg.pt"),
                    help="path to model checkpoint (default: model_slice_pdg.pt)")
    ap.add_argument("--threshold",     type=float, default=0.5)
    ap.add_argument("--all-functions", action="store_true",
                    help="score every function in the IR, sorted by score desc")
    ap.add_argument("--context",       action="store_true",
                    help="print PDG slice vulnerability context alongside each score")
    args = ap.parse_args()

    ir_text = Path(args.ir_file).read_text(errors="replace")
    model, model_type = _load_model(args.model)

    if args.all_functions:
        results = scan_all_fns(ir_text, model, model_type, args.threshold, args.context)
        if not results:
            print("ERROR: no scoreable functions found in IR")
            sys.exit(1)
        for fn_name, n_nodes, prob, ctx in results:
            label = "VULNERABLE" if prob >= args.threshold else "safe"
            print(f"  {fn_name:<40s} [{n_nodes:>4} nodes]  {prob:.1%}  ->  {label}")
            if ctx:
                for line in ctx.splitlines():
                    print(f"    {line}")
                print()
        sys.exit(0)

    # Single-function mode
    if model_type == "slice_pdg":
        from preprocess_slice_pdg import ir_to_graph_slice_pdg
        g = ir_to_graph_slice_pdg(ir_text)
    else:
        g = ir_to_graph(ir_text)

    if g is None:
        print("ERROR: could not parse IR into a graph (no basic blocks found)")
        sys.exit(1)

    n_nodes, prob = _score_graph(g, model, model_type)
    label = "VULNERABLE" if prob >= args.threshold else "safe"

    m = _FN_NAME_RE.search(ir_text[:500])
    fn_name = m.group(1) if m else Path(args.ir_file).stem

    print(f"{Path(args.ir_file).name}  [{n_nodes} nodes]  {prob:.1%}  →  {label}")

    if args.context:
        from preprocess_slice_pdg import ir_to_graph_slice_pdg
        from slice_context import summarize_slice, format_for_llm
        cg = g if model_type == "slice_pdg" else ir_to_graph_slice_pdg(ir_text)
        if cg is not None:
            ctx = format_for_llm(summarize_slice(cg, fn_name=fn_name), score=prob)
            print()
            print(ctx)
        else:
            print("\n[context: could not extract PDG slice]")


if __name__ == "__main__":
    main()
