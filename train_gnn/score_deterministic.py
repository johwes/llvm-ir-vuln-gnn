#!/usr/bin/env python3
"""
score_deterministic.py — Philosophy 2 deterministic ranker + MAX ensemble.

Scores each function using only the structural facts computed by
preprocess_slice_pdg.py + slice_context.py — no trained model, no weights.

Philosophy 2 rule:
  "Does a parameter reach a dangerous sink without a guard?"

Score formula:
  base: n_sinks > 0 AND no guard            → 1.00  (unguarded sink)
        n_sinks > 0 AND null_check only      → 0.75  (weak guard)
        n_sinks > 0 AND bounds_check present → 0.40  (guarded)
        no slice / no sinks                  → 0.05

Multipliers:
  is_external_input   × 1.10
  has_trunc           × 1.05
Score capped at 1.0.

MAX ensemble (--gnn-checkpoint):
  Loads a trained GNN checkpoint and scores each function with it too.
  Final score = max(rule_score, gnn_score).
  Prints rule-only, GNN-only, and MAX ranked tables side by side in summary.

Usage:
    python score_deterministic.py --scarnet --answer-key scarnet-answer-key.txt
    python score_deterministic.py --ir-dir /tmp/scarnet-ir/ --answer-key ...
    python score_deterministic.py --scarnet --answer-key ... \\
        --gnn-checkpoint model_slice_pdg_v8.pt
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from preprocess_slice_pdg import ir_to_graph_slice_pdg
from preprocess_juliet    import ir_to_graph_slice_pdg_v7
from slice_context        import summarize_slice

_SCARNET_REPO = "https://github.com/johwes/scarnet.git"


# ---------------------------------------------------------------------------
# Scoring rule
# ---------------------------------------------------------------------------

def philosophy2_score(summary: dict) -> float:
    """Pure structural Philosophy 2 score from a slice_context summary."""
    n_sinks    = summary["n_sinks"]
    has_guard  = summary["has_guard"]
    guard_type = summary.get("guard_type", "none")
    is_ext     = summary.get("is_external_input", False)
    has_trunc  = summary.get("has_trunc", False)

    if n_sinks == 0:
        base = 0.05
    elif not has_guard:
        base = 1.00
    elif guard_type == "null_check":
        base = 0.75   # null check doesn't protect buffer writes
    else:
        gd = summary.get("guard_density", 1.0)
        if gd == float("inf"):
            base = 1.00
        elif gd >= 5:
            base = 0.70
        elif gd >= 2:
            base = 0.55
        else:
            base = 0.40

    mult = 1.0
    if is_ext:
        mult *= 1.10
    if has_trunc:
        mult *= 1.05

    return min(base * mult, 1.0)


# ---------------------------------------------------------------------------
# GNN scorer (optional — loaded only when --gnn-checkpoint is given)
# ---------------------------------------------------------------------------

def _load_gnn(checkpoint: Path):
    """Load a SlicePDGGNN_v7 checkpoint. Returns (model, preprocess_fn)."""
    from train_slice_pdg_v7 import SlicePDGGNN_v7, VOCAB_SIZE
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    embed_dim = ckpt["embed.weight"].shape[1]
    hidden    = ckpt["lin.weight"].shape[1]
    model     = SlicePDGGNN_v7(VOCAB_SIZE, embed_dim, hidden)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def _gnn_score(model, fn_ir: str, fn_name: str) -> float | None:
    """Score one function with the GNN. Returns sigmoid probability or None."""
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    import numpy as np

    g = ir_to_graph_slice_pdg_v7(fn_ir, fn_name=fn_name)
    if g is None or g.get("x") is None:
        return None

    x          = torch.tensor(g["x"],          dtype=torch.long)
    edge_index = torch.tensor(g["edge_index"],  dtype=torch.long)
    edge_type  = torch.tensor(g["edge_type"],   dtype=torch.long)
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    if x.shape[1] < 3:
        pad = torch.zeros(x.shape[0], 3 - x.shape[1], dtype=torch.long)
        x   = torch.cat([x, pad], dim=1)

    data   = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    batch  = torch.zeros(x.shape[0], dtype=torch.long)

    with torch.no_grad():
        logit = model(data.x, data.edge_index, data.edge_type, batch)
        return float(torch.sigmoid(logit).item())


# ---------------------------------------------------------------------------
# IR utilities
# ---------------------------------------------------------------------------

def _split_functions(ir_text: str) -> list[tuple[str, str]]:
    header_lines = []
    for line in ir_text.splitlines():
        if line.startswith("define"):
            break
        header_lines.append(line)
    header = "\n".join(header_lines)

    segs = re.split(r'(?=^define\b)', ir_text, flags=re.MULTILINE)
    out  = []
    for seg in segs:
        seg = seg.strip()
        if not seg.startswith("define"):
            continue
        m = re.match(r'define\s+.*?@([\w.]+)\s*\(', seg)
        if not m:
            continue
        fn_name = m.group(1)
        fn_ir   = header + "\n\n" + seg + "\n"
        out.append((fn_name, fn_ir))
    return out


def _collect_functions(ir_path: Path) -> list[tuple[str, str]]:
    files = [ir_path] if ir_path.is_file() else sorted(ir_path.glob("**/*.ll"))
    out   = []
    for f in files:
        out.extend(_split_functions(f.read_text(errors="replace")))
    return out


def _load_answer_key(path: Path) -> set[str]:
    return {l.strip() for l in path.read_text().splitlines()
            if l.strip() and not l.startswith("#")}


_SCARNET_SRCS = [
    "src/parse.c",
    "src/handler.c",
    "src/util.c",
    "src/session.c",
    "main.c",
]


def _setup_scarnet_ir(keep_ir: Path | None) -> tuple[Path, Path | None]:
    tmpdir    = Path(tempfile.mkdtemp(prefix="scarnet-det-"))
    clone_dir = tmpdir / "scarnet"
    print(f"Cloning {_SCARNET_REPO} ...")
    subprocess.run(["git", "clone", "--quiet", "--depth=1", _SCARNET_REPO, str(clone_dir)],
                   check=True)
    ir_out = keep_ir if keep_ir else tmpdir / "ir"
    ir_out.mkdir(parents=True, exist_ok=True)
    print(f"Compiling {len(_SCARNET_SRCS)} C file(s) to LLVM IR ...")
    compiled = 0
    for rel in _SCARNET_SRCS:
        cf     = clone_dir / rel
        base   = rel.replace("/", "_").removesuffix(".c")
        out_ll = ir_out / f"{base}.ll"
        result = subprocess.run(
            ["clang-20", "-O0", "-fno-inline", "-S", "-emit-llvm",
             "-I", str(clone_dir / "include"),
             "-w", str(cf), "-o", str(out_ll)],
            capture_output=True)
        if result.returncode == 0:
            compiled += 1
        else:
            print(f"  WARN: {rel} failed to compile")
            if result.stderr:
                print(f"    {result.stderr.decode(errors='replace').strip()[:200]}")
    print(f"  {compiled}/{len(_SCARNET_SRCS)} compiled → {ir_out}")
    return ir_out, (None if keep_ir else tmpdir)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _p_at_k(ranked: list[tuple[str, float]], answer_key: set[str], k: int):
    top  = {fn for fn, _ in ranked[:k]}
    hits = top & answer_key
    prec = len(hits) / k if k else 0.0
    rec  = len(hits) / len(answer_key) if answer_key else 0.0
    return hits, prec, rec


def _print_table(label: str, ranked: list[tuple[str, float]],
                 answer_key: set[str], top_k: int,
                 details: dict[str, str] | None = None):
    print(f"\n=== {label} ===")
    print(f"  {'Rank':>4}  {'Function':<44} {'Score':>6}  {'Vuln?':<5}"
          + ("  Details" if details else ""))
    print(f"  {'----':>4}  {'-'*44} {'------':>6}  {'-----':<5}")
    boundary = False
    for i, (fn, score) in enumerate(ranked, 1):
        if i == top_k + 1 and not boundary:
            print(f"  {'----':>4}  {'-'*44} {'------':>6}  (below top-{top_k})")
            boundary = True
        vuln = "YES" if fn in answer_key else "no"
        det  = f"  {details[fn]}" if details and fn in details else ""
        print(f"  {i:>4}  {fn:<44} {score:>5.1%}  {vuln:<5}{det}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scarnet",  action="store_true",
                     help="Clone johwes/scarnet and compile to IR")
    src.add_argument("--ir-dir",   type=str,
                     help="Directory of pre-compiled .ll files")
    ap.add_argument("--keep-ir",    type=str, default=None)
    ap.add_argument("--answer-key", type=str, required=True)
    ap.add_argument("--top-k",      type=int, default=None)
    ap.add_argument("--gnn-checkpoint", type=str, default=None,
                    help="Optional SlicePDGGNN_v7 .pt file — enables MAX ensemble column")
    ap.add_argument("--verbose",    action="store_true")
    args = ap.parse_args()

    # --- setup IR ---
    tmpdir = None
    if args.scarnet:
        for tool in ("git", "clang-20"):
            if not shutil.which(tool):
                print(f"ERROR: {tool} not found"); sys.exit(1)
        keep = Path(args.keep_ir) if args.keep_ir else None
        ir_path, tmpdir = _setup_scarnet_ir(keep)
    else:
        ir_path = Path(args.ir_dir)

    functions  = _collect_functions(ir_path)
    answer_key = _load_answer_key(Path(args.answer_key))
    top_k      = args.top_k or len(answer_key)
    print(f"Functions found: {len(functions)}")
    print(f"Answer key: {len(answer_key)} known-vulnerable  (top-K = {top_k})")

    # --- load GNN if requested ---
    gnn_model = None
    if args.gnn_checkpoint:
        ckpt_path = Path(args.gnn_checkpoint)
        if not ckpt_path.exists():
            print(f"ERROR: checkpoint not found: {ckpt_path}"); sys.exit(1)
        print(f"Loading GNN: {ckpt_path.name} ...")
        gnn_model = _load_gnn(ckpt_path)

    # --- score each function ---
    rule_scores: dict[str, float]        = {}
    gnn_scores:  dict[str, float | None] = {}
    details:     dict[str, str]          = {}
    no_slice_rule = []
    no_slice_gnn  = []

    for fn_name, fn_ir in functions:
        # Rule score
        g = ir_to_graph_slice_pdg(fn_ir, fn_name=fn_name)
        if g is None or g.get("x") is None:
            rule_scores[fn_name] = 0.05
            details[fn_name]     = "no slice"
            no_slice_rule.append(fn_name)
        else:
            summary              = summarize_slice(g, fn_name=fn_name)
            rule_scores[fn_name] = philosophy2_score(summary)
            ns    = summary["n_sinks"]
            hg    = summary["has_guard"]
            gt    = summary.get("guard_type", "none")
            ext   = "ext" if summary.get("is_external_input") else ""
            trunc = "+trunc" if summary.get("has_trunc") else ""
            sinks = ",".join(sorted({s.get("fn","?") for s in summary["sinks"]}))
            details[fn_name] = (
                f"sinks={ns} guard={'yes('+gt+')' if hg else 'NO'} "
                f"{ext}{trunc} [{sinks}]"
            )
            if args.verbose:
                print(f"  {fn_name}: {summary['natural_language']}")

        # GNN score
        if gnn_model is not None:
            gs = _gnn_score(gnn_model, fn_ir, fn_name)
            gnn_scores[fn_name] = gs
            if gs is None:
                no_slice_gnn.append(fn_name)

    # --- build ranked lists ---
    rule_ranked = sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)
    _print_table("Philosophy 2 rule", rule_ranked, answer_key, top_k, details)
    rule_hits, rule_prec, rule_rec = _p_at_k(rule_ranked, answer_key, top_k)

    gnn_ranked = max_ranked = None
    if gnn_model is not None:
        # GNN-only ranking (treat None as 0.05)
        gnn_ranked = sorted(
            ((fn, gs if gs is not None else 0.05) for fn, gs in gnn_scores.items()),
            key=lambda x: x[1], reverse=True,
        )
        _print_table("GNN only", gnn_ranked, answer_key, top_k)

        # MAX ensemble
        max_scores = {
            fn: max(rule_scores.get(fn, 0.05),
                    gnn_scores.get(fn) if gnn_scores.get(fn) is not None else 0.05)
            for fn in rule_scores
        }
        max_ranked = sorted(max_scores.items(), key=lambda x: x[1], reverse=True)
        _print_table("MAX(rule, GNN)", max_ranked, answer_key, top_k, details)

    # --- summary ---
    print(f"\n{'='*65}")
    print(f"  {'Method':<30} {'Hits':>6}  {'P@K':>6}  {'R@K':>6}")
    print(f"  {'-'*30} {'------':>6}  {'------':>6}  {'------':>6}")
    h, p, r = _p_at_k(rule_ranked, answer_key, top_k)
    print(f"  {'Philosophy 2 rule':<30} {len(h):>3}/{len(answer_key):<2}  {p:>6.1%}  {r:>6.1%}")
    if gnn_ranked is not None:
        h, p, r = _p_at_k(gnn_ranked, answer_key, top_k)
        print(f"  {'GNN only ('+Path(args.gnn_checkpoint).stem+')':<30} {len(h):>3}/{len(answer_key):<2}  {p:>6.1%}  {r:>6.1%}")
    if max_ranked is not None:
        h, p, r = _p_at_k(max_ranked, answer_key, top_k)
        print(f"  {'MAX(rule, GNN)':<30} {len(h):>3}/{len(answer_key):<2}  {p:>6.1%}  {r:>6.1%}")
    print(f"{'='*65}")

    print(f"\n  No-slice (rule): {', '.join(no_slice_rule) or 'none'}")
    if gnn_model is not None:
        print(f"  No-slice (GNN):  {', '.join(no_slice_gnn) or 'none'}")

    rule_misses = sorted(answer_key - {fn for fn, _ in rule_ranked[:top_k]})
    print(f"\n  Rule misses: {rule_misses}")
    if max_ranked is not None:
        max_misses = sorted(answer_key - {fn for fn, _ in max_ranked[:top_k]})
        print(f"  MAX  misses: {max_misses}")

    if tmpdir and tmpdir.exists():
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
