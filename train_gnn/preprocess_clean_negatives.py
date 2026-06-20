#!/usr/bin/env python3
"""
preprocess_clean_negatives.py — §30 clean real-C negatives corpus.

§29 showed that the Juliet-only model saturates on real production code
(91–100% on everything) because it has never seen clean real C — only
synthetic Juliet good functions.  §30 fixes this by adding confirmed-clean
real C functions as negative training examples.

Sources (auto-cloned / downloaded):
  zlib     github.com/madler/zlib          ~50 functions
  musl     github.com/bminor/musl          ~2,000 functions
  SQLite   sqlite.org amalgamation         ~1,500 functions

All three:
  - Compile cleanly via our existing compile_to_ir() pipeline
  - Are heavily audited / security-focused open source projects
  - Cover diverse real C idioms (memory mgmt, string, I/O, math)
  - Are labelled 0 (benign) — the model learns "real clean C != unguarded sink"

Output: data/train_clean_neg_graphs.pkl
        data/valid_clean_neg_graphs.pkl

These are combined with the Juliet positives in train_slice_pdg_v9.py.

Usage:
    python preprocess_clean_negatives.py --workers 4
    python preprocess_clean_negatives.py --subset 500 --workers 1  # smoke test
    python preprocess_clean_negatives.py --skip-clone              # if repos exist
"""

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
SRC  = DATA / "clean_src"

sys.path.insert(0, str(HERE))
from preprocess import compile_to_ir
from preprocess_juliet import ir_to_graph_slice_pdg_v7, _LOCAL_INCLUDE_RE

# ---------------------------------------------------------------------------
# Source repositories
# ---------------------------------------------------------------------------

SOURCES = {
    "zlib": {
        "type": "git",
        "url":  "https://github.com/madler/zlib.git",
        "dir":  SRC / "zlib",
        "glob": "*.c",
    },
    "musl": {
        "type": "git",
        "url":  "https://git.musl-libc.org/git/musl",
        "dir":  SRC / "musl",
        "glob": "src/**/*.c",
    },
    "lua": {
        # ~30 self-contained .c files at repo root, pure C, minimal deps
        "type": "git",
        "url":  "https://github.com/lua/lua.git",
        "dir":  SRC / "lua",
        "glob": "*.c",
    },
    "libuv": {
        # Node.js async I/O library — sockets, buffers, handles; closest in
        # structure to scarnet server functions (parse, session, dispatch)
        "type": "git",
        "url":  "https://github.com/libuv/libuv.git",
        "dir":  SRC / "libuv",
        "glob": "src/*.c",
    },
    "cjson": {
        # Single-file JSON parser — clean application-level C
        "type": "git",
        "url":  "https://github.com/DaveGamble/cJSON.git",
        "dir":  SRC / "cjson",
        "glob": "*.c",
    },
    "lz4": {
        "type": "git",
        "url":  "https://github.com/lz4/lz4.git",
        "dir":  SRC / "lz4",
        "glob": "**/*.c",   # catch lib/ and top-level
    },
}

# musl subdirectories that are most likely to have clean, non-arch-specific code
MUSL_INCLUDE_DIRS = {
    "src/string", "src/stdlib", "src/stdio", "src/malloc",
    "src/math", "src/ctype", "src/network", "src/time",
    "src/unistd", "src/stat", "src/dirent", "src/fcntl",
    "src/internal", "src/env", "src/mman", "src/prng",
    "src/regex", "src/search", "src/temp", "src/passwd",
}


def _clone_git(url: str, dest: Path) -> None:
    if dest.exists() and (dest / ".git").exists():
        print(f"  {dest.name}: already cloned, skipping.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Cloning {url} → {dest} ...")
    subprocess.run(
        ["git", "clone", "--depth=1", url, str(dest)],
        check=True,
    )


def acquire_sources(skip_clone: bool) -> None:
    if skip_clone:
        print("  --skip-clone: assuming sources already present.")
        return
    SRC.mkdir(parents=True, exist_ok=True)
    for name, cfg in SOURCES.items():
        print(f"\n-- {name} --")
        _clone_git(cfg["url"], cfg["dir"])


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def collect_c_files(source_name: str, cfg: dict,
                    max_files: int | None = None) -> list[Path]:
    root = cfg["dir"]
    if not root.exists():
        print(f"  WARNING: {root} does not exist — skipping {source_name}")
        return []

    all_files = sorted(root.glob(cfg["glob"]))

    if source_name == "musl":
        # Keep only well-audited, portable subdirectories
        filtered = []
        for f in all_files:
            rel = f.relative_to(root)
            parts = str(rel.parent)
            if any(parts.startswith(d) for d in MUSL_INCLUDE_DIRS):
                filtered.append(f)
        all_files = filtered

    if max_files:
        all_files = all_files[:max_files]

    print(f"  {source_name}: {len(all_files)} .c files")
    return all_files


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_c_file(args: tuple[str, Path]) -> list[dict]:
    """
    Compile one .c file to IR and extract one graph per function.
    Returns a (possibly empty) list of graph dicts, all labelled y=0.
    """
    source_name, c_path = args
    try:
        src_text = c_path.read_text(errors="replace")
    except Exception:
        return []

    # Strip local #include "..." lines — same technique as preprocess_juliet.py.
    # Project headers (lua.h, uv.h, lz4.h, etc.) are not on the system path;
    # compile_to_ir()'s iterative stub injector handles any remaining unknowns.
    src_text = _LOCAL_INCLUDE_RE.sub("", src_text)

    ir = compile_to_ir(src_text)
    if ir is None:
        return []

    # Extract all non-declaration functions from the IR
    import llvmlite.binding as llvm
    try:
        mod = llvm.parse_assembly(ir)
    except Exception:
        return []

    graphs = []
    for fn in mod.functions:
        if fn.is_declaration:
            continue
        fn_name = fn.name
        if fn_name.startswith("__"):
            continue  # skip internal/compiler-generated stubs

        # Build the per-function IR module (same pattern as eval_all_models.py)
        fn_ir = f"target datalayout = \"\"\ntarget triple = \"x86_64-pc-linux-gnu\"\n\n"
        fn_ir += f"; Function from {source_name}/{c_path.name}\n"
        fn_ir += str(fn) + "\n"

        g = ir_to_graph_slice_pdg_v7(ir, fn_name=fn_name)
        if g is None:
            continue

        g["y"]       = 0  # confirmed clean
        g["fn_name"] = fn_name
        g["source"]  = source_name
        # Remove internal bookkeeping keys
        g.pop("_sliced",  None)
        g.pop("_n_sinks", None)
        graphs.append(g)

    return graphs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--workers",     type=int,  default=4)
    ap.add_argument("--seed",        type=int,  default=42)
    ap.add_argument("--valid-frac",  type=float, default=0.1)
    ap.add_argument("--subset",      type=int,  default=None,
                    help="Limit total files per source (smoke test)")
    ap.add_argument("--skip-clone",  action="store_true",
                    help="Skip git clone / download (sources already present)")
    ap.add_argument("--sources",     type=str,
                    default="zlib,musl,lua,libuv,cjson,lz4",
                    help="Comma-separated sources to use")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    DATA.mkdir(parents=True, exist_ok=True)
    selected_sources = set(args.sources.split(","))

    acquire_sources(args.skip_clone)

    # Collect .c file paths across all sources
    all_tasks: list[tuple[str, Path]] = []
    print()
    for name, cfg in SOURCES.items():
        if name not in selected_sources:
            continue
        files = collect_c_files(name, cfg, args.subset)
        all_tasks.extend((name, f) for f in files)

    rng.shuffle(all_tasks)
    print(f"\nTotal .c files to process: {len(all_tasks)}")

    # Process in parallel
    all_graphs: list[dict] = []
    fail = 0

    print(f"Processing with {args.workers} workers ...")
    if args.workers == 1:
        for i, task in enumerate(all_tasks, 1):
            gs = process_c_file(task)
            all_graphs.extend(gs)
            if i % 50 == 0:
                print(f"  {i}/{len(all_tasks)}  graphs={len(all_graphs)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(process_c_file, t): t for t in all_tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    gs = fut.result()
                    all_graphs.extend(gs)
                except Exception:
                    fail += 1
                if i % 50 == 0:
                    print(f"  {i}/{len(all_tasks)}  graphs={len(all_graphs)}  fail={fail}")

    print(f"\nTotal graphs extracted: {len(all_graphs)}  (from {len(all_tasks)} files, {fail} errors)")

    if not all_graphs:
        print("ERROR: no graphs produced. Check that sources compiled correctly.")
        sys.exit(1)

    # Source breakdown
    from collections import Counter
    src_counts = Counter(g["source"] for g in all_graphs)
    for src, n in sorted(src_counts.items()):
        print(f"  {src}: {n} graphs")

    # Feature stats
    nc = [g["x"].shape[0] for g in all_graphs]
    import statistics
    print(f"  Slice sizes: mean={statistics.mean(nc):.0f}  "
          f"median={statistics.median(nc):.0f}  max={max(nc)}")

    # Check: how many have sinks (non-zero guard features)?
    n_with_sinks = sum(1 for g in all_graphs
                       if g["x"].shape[1] > 1 and np.any(g["x"][:, 1] > 0))
    print(f"  Graphs with guard features (real sinks found): {n_with_sinks} "
          f"({100*n_with_sinks/len(all_graphs):.0f}%)")
    print(f"  (these are clean functions that happen to call dangerous sinks — "
          f"the graph shows a guarded or unguarded pattern but label=0)")

    # Split train/valid
    rng.shuffle(all_graphs)
    n_valid     = max(1, int(len(all_graphs) * args.valid_frac))
    valid_graphs = all_graphs[:n_valid]
    train_graphs = all_graphs[n_valid:]

    train_out = DATA / "train_clean_neg_graphs.pkl"
    valid_out = DATA / "valid_clean_neg_graphs.pkl"

    with open(train_out, "wb") as f:
        pickle.dump(train_graphs, f)
    with open(valid_out, "wb") as f:
        pickle.dump(valid_graphs, f)

    print(f"\nSaved:")
    print(f"  {train_out}  ({len(train_graphs)} graphs)")
    print(f"  {valid_out}  ({len(valid_graphs)} graphs)")
    print(f"\nNext: python train_slice_pdg_v9.py")


if __name__ == "__main__":
    main()
