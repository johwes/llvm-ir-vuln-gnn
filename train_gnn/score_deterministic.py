#!/usr/bin/env python3
"""
score_deterministic.py — Philosophy 2 deterministic ranker.

Scores each function using only the structural facts already computed by
preprocess_slice_pdg.py + slice_context.py — no trained model, no weights.

Philosophy 2 rule:
  "Does a parameter reach a dangerous sink without a guard?"

Score formula:
  base: n_sinks > 0 AND no guard            → 1.00  (unguarded sink)
        n_sinks > 0 AND null_check only      → 0.75  (weak guard — doesn't protect buffer writes)
        n_sinks > 0 AND bounds_check present → 0.40  (guarded — may still be incomplete)
        no slice / no sinks                  → 0.05  (no structural signal)

Multipliers (applied on top of base):
  is_external_input                          × 1.10  (network/user data confirmed)
  has_trunc AND size-taking sink             × 1.05  (integer truncation risk)

Score is capped at 1.0.

Usage:
    # Auto-clone johwes/scarnet
    python score_deterministic.py --scarnet --answer-key scarnet-answer-key.txt

    # Re-use previously compiled IR
    python score_deterministic.py --ir-dir /tmp/scarnet-ir/ --answer-key scarnet-answer-key.txt
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from preprocess_slice_pdg import ir_to_graph_slice_pdg
from slice_context import summarize_slice

_SCARNET_REPO = "https://github.com/johwes/scarnet.git"


# ---------------------------------------------------------------------------
# Scoring rule
# ---------------------------------------------------------------------------

def philosophy2_score(summary: dict) -> float:
    """
    Pure structural score from a slice_context summary dict.

    Returns a float in [0, 1].
    """
    n_sinks   = summary["n_sinks"]
    has_guard = summary["has_guard"]
    guard_type = summary.get("guard_type", "none")
    is_ext    = summary.get("is_external_input", False)
    has_trunc = summary.get("has_trunc", False)

    if n_sinks == 0:
        base = 0.05
    elif not has_guard:
        base = 1.00   # unguarded sink — canonical Philosophy 2 hit
    elif guard_type == "null_check":
        base = 0.75   # null check present but no bounds check — weak protection for buffer sinks
    else:
        # bounds_check or mixed — some guard present
        gd = summary.get("guard_density", 1.0)
        if gd == float("inf"):
            base = 1.00
        elif gd >= 5:
            base = 0.70   # very sparse: many sinks per guard
        elif gd >= 2:
            base = 0.55   # sparse
        else:
            base = 0.40   # well-covered

    mult = 1.0
    if is_ext:
        mult *= 1.10
    if has_trunc:
        mult *= 1.05

    return min(base * mult, 1.0)


# ---------------------------------------------------------------------------
# IR utilities (minimal copies from eval_all_models.py)
# ---------------------------------------------------------------------------

def _split_functions(ir_text: str) -> list[tuple[str, str]]:
    """Split a .ll module into (fn_name, fn_ir) pairs."""
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
    """Collect (fn_name, fn_ir) from a directory of .ll files or a single file."""
    if ir_path.is_file():
        files = [ir_path]
    else:
        files = sorted(ir_path.glob("**/*.ll"))

    out = []
    for f in files:
        ir_text = f.read_text(errors="replace")
        out.extend(_split_functions(ir_text))
    return out


def _load_answer_key(path: Path) -> set[str]:
    lines = path.read_text().splitlines()
    return {l.strip() for l in lines if l.strip() and not l.startswith("#")}


def _setup_scarnet_ir(keep_ir: Path | None) -> tuple[Path, Path | None]:
    tmpdir    = Path(tempfile.mkdtemp(prefix="scarnet-det-"))
    clone_dir = tmpdir / "scarnet"

    print(f"Cloning {_SCARNET_REPO} ...")
    subprocess.run(["git", "clone", "--depth=1", _SCARNET_REPO, str(clone_dir)],
                   check=True, capture_output=True)

    ir_out = keep_ir if keep_ir else tmpdir / "ir"
    ir_out.mkdir(parents=True, exist_ok=True)

    c_files = sorted(clone_dir.glob("**/*.c"))
    print(f"Compiling {len(c_files)} C file(s) to LLVM IR ...")
    for cf in c_files:
        out_ll = ir_out / (cf.stem + ".ll")
        result = subprocess.run(
            ["clang-20", "-O0", "-fno-inline", "-S", "-emit-llvm",
             "-w", str(cf), "-o", str(out_ll)],
            capture_output=True)
        if result.returncode != 0:
            print(f"  WARN: {cf.name} failed to compile")

    return ir_out, (None if keep_ir else tmpdir)


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
    ap.add_argument("--keep-ir",   type=str, default=None,
                    help="Save compiled IR here (with --scarnet)")
    ap.add_argument("--answer-key", type=str, required=True,
                    help="Plain-text file: one known-vulnerable fn name per line")
    ap.add_argument("--top-k",     type=int, default=None,
                    help="Evaluate top-K precision/recall (default = len(answer_key))")
    ap.add_argument("--verbose",   action="store_true",
                    help="Print per-function slice details")
    args = ap.parse_args()

    # --- setup IR ---
    tmpdir = None
    if args.scarnet:
        for tool in ("git", "clang-20"):
            if not shutil.which(tool):
                print(f"ERROR: {tool} not found in PATH")
                sys.exit(1)
        keep = Path(args.keep_ir) if args.keep_ir else None
        ir_path, tmpdir = _setup_scarnet_ir(keep)
    else:
        ir_path = Path(args.ir_dir)

    functions = _collect_functions(ir_path)
    print(f"Functions found: {len(functions)}")

    answer_key = _load_answer_key(Path(args.answer_key))
    top_k = args.top_k or len(answer_key)
    print(f"Answer key: {len(answer_key)} known-vulnerable  (top-K = {top_k})\n")

    # --- score each function ---
    results = []
    no_slice = []
    for fn_name, fn_ir in functions:
        g = ir_to_graph_slice_pdg(fn_ir, fn_name=fn_name)
        if g is None or g.get("x") is None:
            no_slice.append(fn_name)
            results.append((fn_name, 0.05, None))
            continue
        summary = summarize_slice(g, fn_name=fn_name)
        score   = philosophy2_score(summary)
        results.append((fn_name, score, summary))

    results.sort(key=lambda r: r[1], reverse=True)

    # --- print ranked table ---
    print(f"=== Philosophy 2 deterministic ranker ===")
    print(f"  {'Rank':>4}  {'Function':<44} {'Score':>6}  {'Vuln?':<5}  Details")
    print(f"  {'----':>4}  {'-'*44} {'------':>6}  {'-----':<5}  -------")
    boundary_printed = False
    for i, (fn_name, score, summary) in enumerate(results, 1):
        if i == top_k + 1 and not boundary_printed:
            print(f"  {'----':>4}  {'-'*44} {'------':>6}  {'(below top-%d)' % top_k}")
            boundary_printed = True
        vuln = "YES" if fn_name in answer_key else "no"

        if summary is None:
            detail = "no slice"
        else:
            ns    = summary["n_sinks"]
            hg    = summary["has_guard"]
            gt    = summary.get("guard_type", "none")
            ext   = "ext" if summary.get("is_external_input") else ""
            trunc = "+trunc" if summary.get("has_trunc") else ""
            sinks = ",".join(sorted({s.get("fn","?") for s in summary["sinks"]}))
            detail = f"sinks={ns} guard={'yes('+gt+')' if hg else 'NO'} {ext}{trunc} [{sinks}]"

        print(f"  {i:>4}  {fn_name:<44} {score:>5.1%}  {vuln:<5}  {detail}")

        if args.verbose and summary:
            print(f"         {summary['natural_language']}")

    # --- precision / recall ---
    top_fns = {fn for fn, _, _ in results[:top_k]}
    hits    = top_fns & answer_key
    prec    = len(hits) / top_k if top_k else 0.0
    rec     = len(hits) / len(answer_key) if answer_key else 0.0

    print(f"\n{'='*60}")
    print(f"  Philosophy 2 deterministic  —  top-{top_k} of {len(results)}")
    print(f"  Hits:       {len(hits)}/{len(answer_key)}")
    print(f"  Precision:  {prec:.1%}")
    print(f"  Recall:     {rec:.1%}")
    print(f"  No-slice:   {len(no_slice)} function(s): {', '.join(no_slice) or 'none'}")
    print(f"{'='*60}")

    print(f"\nMisses: {sorted(answer_key - top_fns)}")
    print(f"False positives: {sorted(top_fns - answer_key)}")

    # cleanup
    if tmpdir and tmpdir.exists():
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
