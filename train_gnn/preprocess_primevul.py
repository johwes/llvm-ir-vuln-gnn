#!/usr/bin/env python3
"""
preprocess_primevul.py — Build PDG slice graphs from the PrimeVul dataset.

PrimeVul (ICSE 2025, arXiv:2403.18624) applies LLM-assisted relabeling to
C/C++ functions from real CVE commits, achieving function-level label accuracy
~3x better than Devign/BigVul's commit-level approach.  Using it as a training
source directly targets the 55-58% Devign accuracy ceiling caused by ~10-20%
label noise.

Dataset: https://huggingface.co/datasets/colin/PrimeVul
         ~7k vulnerable + ~229k benign C/C++ functions, 140+ CWEs

Class imbalance: ~1:33 (vuln:benign).  Use --max-benign to cap benign samples
and keep pos_weight manageable.  Recommended: --max-benign 21000 (3:1 ratio).

Usage:
    pip install datasets
    python preprocess_primevul.py --subset 2000              # smoke test
    python preprocess_primevul.py --max-benign 21000         # balanced run
    python preprocess_primevul.py --max-benign 21000 --workers 8
    python preprocess_primevul.py                            # full (slow)

Output: data/{train,valid,test}_primevul_graphs.pkl
  Same format as _slice_pdg_graphs.pkl — drop-in for train_slice_pdg_v5.py.
"""

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"

sys.path.insert(0, str(HERE))
from preprocess import compile_to_ir
from preprocess_slice_pdg import ir_to_graph_slice_pdg


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def _load_from_hf() -> list[dict]:
    """Download PrimeVul from HuggingFace. Returns list of {func, target, ...}."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: HuggingFace datasets not installed.")
        print("       Run: pip install datasets")
        sys.exit(1)

    print("Downloading PrimeVul from HuggingFace (colin/PrimeVul) ...")
    try:
        ds = load_dataset("colin/PrimeVul", trust_remote_code=True)
    except Exception as e:
        print(f"ERROR loading colin/PrimeVul: {e}")
        print("Trying starsofchance/PrimeVul ...")
        try:
            ds = load_dataset("starsofchance/PrimeVul", trust_remote_code=True)
        except Exception as e2:
            print(f"ERROR: {e2}")
            sys.exit(1)

    items = []
    for split_name in ds.keys():
        for i, row in enumerate(ds[split_name]):
            items.append({
                "func":    row["func"],
                "target":  int(row["target"]),
                "idx":     len(items),
                "cve":     row.get("cve", ""),
                "cwe":     row.get("cwe", []),
                "project": row.get("project", ""),
            })
        print(f"  {split_name}: {len(ds[split_name])} items")

    print(f"  Total: {len(items)} items loaded")
    return items


def _split_items(items: list[dict], seed: int = 42) -> tuple[list, list, list]:
    """Stratified 80/10/10 train/valid/test split."""
    rng = random.Random(seed)
    vuln  = [x for x in items if x["target"] == 1]
    benign = [x for x in items if x["target"] == 0]
    rng.shuffle(vuln)
    rng.shuffle(benign)

    def _split(lst):
        n = len(lst)
        n_valid = max(1, int(n * 0.10))
        n_test  = max(1, int(n * 0.10))
        return lst[n_valid + n_test:], lst[:n_valid], lst[n_valid:n_valid + n_test]

    tr_v, va_v, te_v = _split(vuln)
    tr_b, va_b, te_b = _split(benign)

    train = tr_v + tr_b
    valid = va_v + va_b
    test  = te_v + te_b
    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test


# ---------------------------------------------------------------------------
# Per-item processing (mirrors preprocess_slice_pdg.py)
# ---------------------------------------------------------------------------

def process_item(item: dict) -> dict | None:
    ir = compile_to_ir(item["func"])
    if ir is None:
        return None
    g = ir_to_graph_slice_pdg(ir)
    if g is None:
        return None
    g["y"]       = int(item["target"])
    g["idx"]     = item.get("idx", 0)
    g["cve"]     = item.get("cve", "")
    g["project"] = item.get("project", "")
    return g


def process_split(items: list[dict], workers: int, label: str) -> list[dict]:
    graphs, ok, fail = [], 0, 0
    total = len(items)
    print(f"  Processing {total} functions with {workers} worker(s) ...")

    if workers == 1:
        for i, item in enumerate(items, 1):
            g = process_item(item)
            if g:
                graphs.append(g); ok += 1
            else:
                fail += 1
            if i % 500 == 0:
                print(f"    {i}/{total}  ok={ok}  failed={fail}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(process_item, it): it for it in items}
            for i, fut in enumerate(as_completed(futs), 1):
                g = fut.result()
                if g:
                    graphs.append(g); ok += 1
                else:
                    fail += 1
                if i % 500 == 0:
                    print(f"    {i}/{total}  ok={ok}  failed={fail}")

    attrition = fail / total * 100 if total > 0 else 0
    n_vuln  = sum(1 for g in graphs if g["y"] == 1)
    n_benign = len(graphs) - n_vuln
    node_counts = [g["x"].shape[0] for g in graphs]
    n_sliced = sum(1 for g in graphs if g.get("_sliced", False))

    print(f"  Done [{label}]: {ok} graphs  "
          f"({n_vuln} vuln / {n_benign} benign)  "
          f"{fail} failed ({attrition:.0f}%)")
    if node_counts:
        print(f"  Slice sizes: mean={np.mean(node_counts):.0f}  "
              f"median={int(np.median(node_counts))}  max={max(node_counts)}")
        print(f"  Sliced: {n_sliced}/{ok} ({100*n_sliced/ok:.0f}%)")

    for g in graphs:
        g.pop("_sliced", None)
        g.pop("_n_sinks", None)

    return graphs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    ap.add_argument("--subset",     type=int, default=None,
                    help="Total items to use (balanced vuln/benign). Smoke test.")
    ap.add_argument("--max-benign", type=int, default=None,
                    help="Cap benign samples (e.g. 21000 for 3:1 ratio). "
                         "Recommended when training on full dataset.")
    ap.add_argument("--workers",    type=int, default=4)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--cache",      type=str, default=None,
                    help="Path to cached primevul.jsonl (skip HF download)")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)

    # -- Load items -------------------------------------------------------
    cache_path = Path(args.cache) if args.cache else DATA / "primevul_raw.jsonl"

    if cache_path.exists():
        print(f"Loading cached PrimeVul from {cache_path} ...")
        with open(cache_path) as f:
            items = [json.loads(l) for l in f]
        print(f"  {len(items)} items")
    else:
        items = _load_from_hf()
        print(f"Caching to {cache_path} ...")
        with open(cache_path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    vuln   = [x for x in items if x["target"] == 1]
    benign = [x for x in items if x["target"] == 0]
    print(f"\nDataset: {len(vuln)} vulnerable  {len(benign)} benign  "
          f"(ratio 1:{len(benign)//max(len(vuln),1)})")

    # -- Apply limits -------------------------------------------------------
    rng = random.Random(args.seed)

    if args.subset:
        n = args.subset // 2
        rng.shuffle(vuln); rng.shuffle(benign)
        vuln   = vuln[:n]
        benign = benign[:n]
        print(f"--subset: keeping {len(vuln)} vuln + {len(benign)} benign")
    elif args.max_benign and len(benign) > args.max_benign:
        rng.shuffle(benign)
        benign = benign[:args.max_benign]
        print(f"--max-benign: capped benign to {len(benign)} "
              f"(ratio 1:{len(benign)//max(len(vuln),1)})")

    items = vuln + benign

    # -- Split ---------------------------------------------------------------
    train, valid, test = _split_items(items, seed=args.seed)
    print(f"\nSplit: train={len(train)}  valid={len(valid)}  test={len(test)}")
    for label, split in [("train", train), ("valid", valid), ("test", test)]:
        nv = sum(1 for x in split if x["target"] == 1)
        print(f"  {label}: {nv} vuln / {len(split)-nv} benign")

    # -- Process each split --------------------------------------------------
    for split_name, split_items in [("train", train), ("valid", valid), ("test", test)]:
        dst = DATA / f"{split_name}_primevul_graphs.pkl"
        print(f"\n-- {split_name} ({len(split_items)} items) --")
        graphs = process_split(split_items, workers=args.workers, label=split_name)
        with open(dst, "wb") as f:
            pickle.dump(graphs, f)
        print(f"  Saved {len(graphs)} graphs -> {dst}")

    print("\nDone. Run train_slice_pdg_v5.py next.\n")


if __name__ == "__main__":
    main()
