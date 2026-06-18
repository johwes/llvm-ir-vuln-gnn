#!/usr/bin/env python3
"""
preprocess_primevul_joern.py — Build §26 PDG slice graphs from PrimeVul via Joern.

Unlike preprocess_primevul.py (clang pipeline, ~34% coverage), this uses Joern's
fuzzy island-grammar parser: no compilation, ~95% coverage, no header dependencies.

Critical improvement: the clang pipeline sees 2x higher attrition for vulnerable
functions (21.8% survive) vs benign (37% survive), systematically biasing training
toward simple safe functions. Joern parses incomplete types and undefined structs,
recovering the complex stateful functions where Heartbleed-class bugs live.

Output: data/{train,valid,test}_joern_graphs.pkl
  Same format as *_primevul_graphs.pkl — drop-in for train_slice_pdg_v6.py.
  NOTE: VOCAB_SIZE=16 (Joern CPG tokens), not 110 (LLVM opcodes).
        Not compatible with LLVM IR model checkpoints.

Usage:
    python preprocess_primevul_joern.py --subset 500          # smoke test
    python preprocess_primevul_joern.py --max-benign 21000    # balanced run
    python preprocess_primevul_joern.py --max-benign 21000 --workers 8
"""

import argparse
import json
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"

sys.path.insert(0, str(HERE))
from preprocess_joern import ir_to_graph_joern


# ---------------------------------------------------------------------------
# Dataset loading — identical to preprocess_primevul.py
# ---------------------------------------------------------------------------

def _load_from_hf() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: HuggingFace datasets not installed. Run: pip install datasets")
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
            print(f"ERROR: {e2}"); sys.exit(1)

    items = []
    for split_name in ds.keys():
        for row in ds[split_name]:
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
    rng = random.Random(seed)
    vuln   = [x for x in items if x["target"] == 1]
    benign = [x for x in items if x["target"] == 0]
    rng.shuffle(vuln); rng.shuffle(benign)

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
    rng.shuffle(train); rng.shuffle(valid); rng.shuffle(test)
    return train, valid, test


# ---------------------------------------------------------------------------
# Per-item processing (Joern, no compilation)
# ---------------------------------------------------------------------------

def process_item(item: dict) -> dict | None:
    g = ir_to_graph_joern(item["func"])
    if g is None:
        return None
    g["y"]       = int(item["target"])
    g["idx"]     = item.get("idx", 0)
    g["cve"]     = item.get("cve", "")
    g["project"] = item.get("project", "")
    return g


def process_split(items: list[dict], workers: int, label: str) -> list[dict]:
    graphs = []
    ok_vuln = ok_benign = fail_vuln = fail_benign = 0
    total = len(items)
    print(f"  Processing {total} functions with {workers} worker(s) ...")

    def _record(g, item):
        nonlocal ok_vuln, ok_benign, fail_vuln, fail_benign
        is_vuln = item["target"] == 1
        if g:
            graphs.append(g)
            if is_vuln: ok_vuln   += 1
            else:        ok_benign += 1
        else:
            if is_vuln: fail_vuln   += 1
            else:        fail_benign += 1

    if workers == 1:
        for i, item in enumerate(items, 1):
            _record(process_item(item), item)
            if i % 500 == 0:
                print(f"    {i}/{total}  ok={ok_vuln+ok_benign}  "
                      f"fail_v={fail_vuln}  fail_b={fail_benign}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(process_item, it): it for it in items}
            for i, fut in enumerate(as_completed(futs), 1):
                item = futs[fut]
                _record(fut.result(), item)
                if i % 500 == 0:
                    print(f"    {i}/{total}  ok={ok_vuln+ok_benign}  "
                          f"fail_v={fail_vuln}  fail_b={fail_benign}")

    n_ok   = ok_vuln + ok_benign
    n_fail = fail_vuln + fail_benign
    n_vuln_in  = ok_vuln + fail_vuln
    n_benign_in = ok_benign + fail_benign

    vuln_surv   = ok_vuln   / n_vuln_in   * 100 if n_vuln_in   > 0 else 0
    benign_surv = ok_benign / n_benign_in * 100 if n_benign_in > 0 else 0

    node_counts = [g["x"].shape[0] for g in graphs]
    n_sliced    = sum(1 for g in graphs if g.get("_sliced", False))

    print(f"  Done [{label}]: {n_ok} graphs  ({ok_vuln} vuln / {ok_benign} benign)  "
          f"{n_fail} failed")
    print(f"  Survival:  vuln={vuln_surv:.1f}%  benign={benign_surv:.1f}%  "
          f"bias={benign_surv/vuln_surv:.2f}x  "
          f"(clang baseline: vuln=21.8% benign=37.0% bias=1.70x)")
    if node_counts:
        print(f"  Slice sizes: mean={np.mean(node_counts):.0f}  "
              f"median={int(np.median(node_counts))}  max={max(node_counts)}")
        print(f"  Sliced: {n_sliced}/{n_ok} ({100*n_sliced/n_ok:.0f}%)")

    for g in graphs:
        g.pop("_sliced", None)
        g.pop("_fn_name", None)

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
                    help="Total items (balanced vuln/benign). Smoke test.")
    ap.add_argument("--max-benign", type=int, default=None,
                    help="Cap benign samples (e.g. 21000 for ~1:3 ratio).")
    ap.add_argument("--workers",    type=int, default=4)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--cache",      type=str, default=None,
                    help="Path to cached primevul.jsonl (skip HF download)")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)

    # -- Load items -----------------------------------------------------------
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

    # -- Apply limits ---------------------------------------------------------
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

    # -- Split ----------------------------------------------------------------
    train, valid, test = _split_items(items, seed=args.seed)
    print(f"\nSplit: train={len(train)}  valid={len(valid)}  test={len(test)}")
    for lbl, split in [("train", train), ("valid", valid), ("test", test)]:
        nv = sum(1 for x in split if x["target"] == 1)
        print(f"  {lbl}: {nv} vuln / {len(split)-nv} benign")

    # -- Process each split ---------------------------------------------------
    for split_name, split_items in [("train", train), ("valid", valid), ("test", test)]:
        dst = DATA / f"{split_name}_joern_graphs.pkl"
        print(f"\n-- {split_name} ({len(split_items)} items) --")
        graphs = process_split(split_items, workers=args.workers, label=split_name)
        with open(dst, "wb") as f:
            pickle.dump(graphs, f)
        print(f"  Saved {len(graphs)} graphs -> {dst}")

    print("\nDone. Run train_slice_pdg_v6.py next.\n")


if __name__ == "__main__":
    main()
