#!/usr/bin/env python3
"""
preprocess_instr_bigvul.py — Extract paired (vuln, fix) instruction-level IR graphs from BigVul.

Identical to preprocess_bigvul.py but uses ir_to_graph_instr() (each instruction
is a node, ~300-500 nodes/function) instead of ir_to_graph() (block-level, ~15 nodes).

At instruction granularity a 3-line patch changes 3-5 nodes directly, making
(vuln, fix) pairs structurally distinguishable — the key hypothesis for whether
triplet loss can succeed where it failed at block level (pair-sim 0.979).

Outputs: data/bigvul_{train,valid,test}_instr_pairs.pkl

Usage:
    python preprocess_instr_bigvul.py --csv data/MSR_data_cleaned.csv --workers 4
    python preprocess_instr_bigvul.py --csv data/MSR_data_cleaned.csv --subset 500
"""
import argparse
import pickle
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import time
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"

sys.path.insert(0, str(HERE))
from preprocess import compile_to_ir
from preprocess_instr import ir_to_graph_instr


def process_pair(args: tuple) -> dict | None:
    idx, vuln_src, fix_src, cwe, cve = args
    vuln_ir = compile_to_ir(vuln_src)
    if vuln_ir is None:
        return None
    fix_ir = compile_to_ir(fix_src)
    if fix_ir is None:
        return None
    vuln_g = ir_to_graph_instr(vuln_ir)
    fix_g  = ir_to_graph_instr(fix_ir)
    if vuln_g is None or fix_g is None:
        return None
    return {"vuln": vuln_g, "fix": fix_g, "cwe": str(cwe), "cve": str(cve), "idx": idx}


def load_pairs(csv_path: Path, subset: int | None, seed: int) -> list[tuple]:
    size_mb = csv_path.stat().st_size / 1e6
    print(f"Loading {csv_path} ({size_mb:.0f} MB) ...")
    if size_mb > 500:
        print(f"  Large file -- pandas python engine may take several minutes ...")
    t0 = time.time()
    df = pd.read_csv(csv_path, encoding="latin-1", engine="python", on_bad_lines="skip")
    print(f"  Read in {time.time()-t0:.0f}s")
    print(f"  Raw rows: {len(df):,}")

    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for col in df.columns:
        if col.lower() in ("cve id", "cve_id"):   rename[col] = "CVE ID"
        if col.lower() in ("cwe id", "cwe_id"):   rename[col] = "CWE ID"
    if rename:
        df = df.rename(columns=rename)

    needed = ["func_before", "func_after", "CVE ID", "CWE ID", "vul"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}")
        print(f"  Available: {list(df.columns)}")
        sys.exit(1)

    df = df.dropna(subset=["func_before", "func_after"])
    df = df[df["func_before"].str.strip() != ""]
    df = df[df["func_after"].str.strip()  != ""]
    df = df[df["func_before"] != df["func_after"]]
    df = df[df["vul"] == 1]
    df = df.drop_duplicates(subset=["func_before", "func_after"])
    print(f"  Usable pairs (vul=1, before!=after): {len(df):,}")

    rng  = random.Random(seed)
    idxs = list(df.index)
    rng.shuffle(idxs)
    if subset:
        idxs = idxs[:subset]
        print(f"  Using subset of {len(idxs):,} pairs")

    result = []
    for idx in idxs:
        row = df.loc[idx]
        cwe = str(row["CWE ID"]) if pd.notna(row["CWE ID"]) else "CWE-unknown"
        cve = str(row["CVE ID"]) if pd.notna(row["CVE ID"]) else "unknown"
        result.append((
            idx,
            row["func_before"],
            row["func_after"],
            cwe or "CWE-unknown",
            cve or "unknown",
        ))
    return result


def split_by_cve(pairs: list[tuple], seed: int) -> dict[str, list[tuple]]:
    cve_to_pairs: dict[str, list] = defaultdict(list)
    for p in pairs:
        cve_to_pairs[p[4]].append(p)

    cves = list(cve_to_pairs.keys())
    random.Random(seed).shuffle(cves)
    n = len(cves)
    train_cves = set(cves[:int(n * 0.8)])
    valid_cves = set(cves[int(n * 0.8):int(n * 0.9)])

    splits: dict[str, list] = {"train": [], "valid": [], "test": []}
    for cve, ps in cve_to_pairs.items():
        key = "train" if cve in train_cves else "valid" if cve in valid_cves else "test"
        splits[key].extend(ps)

    for name, ps in splits.items():
        print(f"  {name}: {len(ps):,} pairs from {len({p[4] for p in ps}):,} CVEs")
    return splits


def process_split(pairs: list[tuple], workers: int) -> list[dict]:
    graphs: list[dict] = []
    ok = fail = 0
    total = len(pairs)
    print(f"  Compiling {total:,} pairs with {workers} worker(s) ...")

    if workers == 1:
        for i, args in enumerate(pairs, 1):
            g = process_pair(args)
            if g: graphs.append(g); ok += 1
            else: fail += 1
            if i % max(1, min(50, total//10)) == 0:
                print(f"    {i}/{total}  ok={ok}  failed={fail}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(process_pair, a): a for a in pairs}
            for i, fut in enumerate(as_completed(futs), 1):
                g = fut.result()
                if g: graphs.append(g); ok += 1
                else: fail += 1
                if i % 200 == 0:
                    print(f"    {i}/{total}  ok={ok}  failed={fail}")

    attrition = fail/total*100 if total > 0 else 0
    print(f"  Done: {ok} pairs compiled, {fail} failed ({attrition:.0f}% attrition)")
    return graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",     default=str(DATA / "MSR_data_cleaned.csv"))
    ap.add_argument("--subset",  type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("  Download: gdown 1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X -O data/bigvul.zip")
        print("  Then: unzip data/bigvul.zip -d data/")
        sys.exit(1)

    DATA.mkdir(exist_ok=True)
    pairs = load_pairs(csv_path, args.subset, args.seed)
    print(f"\nSplitting {len(pairs):,} pairs by CVE ...")
    splits = split_by_cve(pairs, args.seed)

    for split_name, split_pairs in splits.items():
        print(f"\n-- {split_name} ------------------------------------------------")
        graphs = process_split(split_pairs, args.workers)
        out = DATA / f"bigvul_{split_name}_instr_pairs.pkl"
        with open(out, "wb") as f:
            pickle.dump(graphs, f)
        print(f"  Saved {len(graphs):,} graphs -> {out}")

    print("\nDone. Run train_instr_triplet.py next.\n")


if __name__ == "__main__":
    main()
