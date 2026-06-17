#!/usr/bin/env python3
"""
preprocess_bigvul_cls.py — §21 BigVul binary classifier preprocessing.

Produces instruction-level graphs in the v2/§13 format (Perfograph + categorical
call targets), one graph per function, with binary labels (1=vulnerable, 0=clean).

Three sources of examples:
  positive  (label=1): func_before where vul=1  — the vulnerable version
  negative1 (label=0): func_after  where vul=1  — the patch (same CVE, fixed code)
  negative2 (label=0): func_before where vul=0  — unrelated clean functions

Split strategy:
  vul=1 pairs: split 80/10/10 by CVE ID (no data leakage across CVEs)
  vul=0 items: split 80/10/10 randomly (no CVE to leak)

Outputs:
  data/bigvul_cls_{train,valid,test}_instr_v2_graphs.pkl
  Each graph dict: {"x": float32 (N,2), "edge_index": int64 (2,E),
                    "edge_type": int64 (E,), "y": int, "cwe": str, "idx": int}

Usage:
    python preprocess_bigvul_cls.py --csv data/MSR_data_cleaned.csv
    python preprocess_bigvul_cls.py --subset 300    # smoke test (~200 graphs)
    python preprocess_bigvul_cls.py --workers 8
"""
import argparse
import pickle
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"

sys.path.insert(0, str(HERE))
from preprocess import compile_to_ir
from preprocess_instr_v2 import ir_to_graph_instr


def _worker(args: tuple) -> dict | None:
    src, label, cwe, idx = args
    ir = compile_to_ir(src)
    if ir is None:
        return None
    g = ir_to_graph_instr(ir)
    if g is None:
        return None
    g["y"]   = label
    g["cwe"] = cwe
    g["idx"] = idx
    return g


def load_items(csv_path: Path, subset: int | None, seed: int
               ) -> tuple[list[tuple], list[tuple]]:
    """Return (vuln_pairs, clean_singletons) from the BigVul CSV."""
    size_mb = csv_path.stat().st_size / 1e6
    print(f"Loading {csv_path} ({size_mb:.0f} MB) ...")
    if size_mb > 500:
        print("  Large file — this may take several minutes ...")
    t0 = time.time()
    df = pd.read_csv(csv_path, encoding="latin-1", engine="python", on_bad_lines="skip")
    print(f"  Read in {time.time()-t0:.0f}s  ({len(df):,} rows)")

    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for col in df.columns:
        if col.lower() in ("cve id", "cve_id"):  rename[col] = "CVE ID"
        if col.lower() in ("cwe id", "cwe_id"):  rename[col] = "CWE ID"
    if rename:
        df = df.rename(columns=rename)

    needed = ["func_before", "func_after", "CVE ID", "CWE ID", "vul"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}")
        print(f"  Available: {list(df.columns)}")
        sys.exit(1)

    df = df.dropna(subset=["func_before"])
    df = df[df["func_before"].str.strip() != ""]

    # Filter to C source files only — BigVul covers multiple languages;
    # compile_to_ir only handles C so trying others wastes time and inflates attrition.
    before = len(df)
    if "file_name" in df.columns:
        df = df[df["file_name"].str.lower().str.endswith(".c", na=False)]
        print(f"  C-only filter (file_name ends in .c): {len(df):,} rows kept "
              f"(dropped {before - len(df):,})")
    elif "lang" in df.columns:
        df = df[df["lang"].str.strip().str.upper() == "C"]
        print(f"  C-only filter (lang==C): {len(df):,} rows kept "
              f"(dropped {before - len(df):,})")
    else:
        print("  WARNING: no file_name or lang column found — skipping C filter "
              "(expect high attrition from non-C functions)")

    # --- vul=1 pairs: before (label=1) and after (label=0) ----------------------
    vuln_df = df[df["vul"] == 1].copy()
    vuln_df = vuln_df.dropna(subset=["func_after"])
    vuln_df = vuln_df[vuln_df["func_after"].str.strip() != ""]
    vuln_df = vuln_df[vuln_df["func_before"] != vuln_df["func_after"]]
    vuln_df = vuln_df.drop_duplicates(subset=["func_before"])

    # --- vul=0 clean functions: label=0 -----------------------------------------
    clean_df = df[df["vul"] == 0].copy()
    clean_df = clean_df.drop_duplicates(subset=["func_before"])

    print(f"  vul=1 pairs (before!=after): {len(vuln_df):,}")
    print(f"  vul=0 clean functions:        {len(clean_df):,}")

    rng = random.Random(seed)
    vuln_idxs  = list(vuln_df.index)
    clean_idxs = list(clean_df.index)
    rng.shuffle(vuln_idxs)
    rng.shuffle(clean_idxs)

    if subset:
        vuln_idxs  = vuln_idxs[:subset]
        clean_idxs = clean_idxs[:subset]
        print(f"  Subset: {len(vuln_idxs):,} vuln pairs + {len(clean_idxs):,} clean")

    pairs = []
    for idx in vuln_idxs:
        row = vuln_df.loc[idx]
        cwe = str(row["CWE ID"]) if pd.notna(row["CWE ID"]) else "CWE-unknown"
        cve = str(row["CVE ID"]) if pd.notna(row["CVE ID"]) else "unknown"
        pairs.append((int(idx), str(row["func_before"]), str(row["func_after"]), cwe, cve))

    cleans = []
    for idx in clean_idxs:
        row = clean_df.loc[idx]
        cwe = str(row["CWE ID"]) if pd.notna(row["CWE ID"]) else "CWE-unknown"
        cleans.append((int(idx), str(row["func_before"]), cwe))

    return pairs, cleans


def _split_by_cve(pairs: list[tuple], seed: int) -> dict[str, list[tuple]]:
    cve_to: dict[str, list] = defaultdict(list)
    for p in pairs:
        cve_to[p[4]].append(p)
    cves = list(cve_to.keys())
    random.Random(seed).shuffle(cves)
    n = len(cves)
    train_cves = set(cves[:int(n * 0.8)])
    valid_cves = set(cves[int(n * 0.8):int(n * 0.9)])
    splits: dict[str, list] = {"train": [], "valid": [], "test": []}
    for cve, ps in cve_to.items():
        key = "train" if cve in train_cves else "valid" if cve in valid_cves else "test"
        splits[key].extend(ps)
    return splits


def _split_random(items: list, seed: int,
                  train_frac: float = 0.8, valid_frac: float = 0.1) -> dict[str, list]:
    lst = list(items)
    random.Random(seed + 1).shuffle(lst)
    n = len(lst)
    t = int(n * train_frac)
    v = int(n * (train_frac + valid_frac))
    return {"train": lst[:t], "valid": lst[t:v], "test": lst[v:]}


def compile_split(work: list[tuple], workers: int) -> list[dict]:
    graphs: list[dict] = []
    ok = fail = 0
    total = len(work)
    print(f"  Compiling {total:,} items with {workers} worker(s) ...")

    if workers == 1:
        for i, args in enumerate(work, 1):
            g = _worker(args)
            if g:
                graphs.append(g); ok += 1
            else:
                fail += 1
            if i % max(1, min(100, total // 10)) == 0:
                print(f"    {i}/{total}  ok={ok}  fail={fail}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_worker, a): a for a in work}
            for i, fut in enumerate(as_completed(futs), 1):
                g = fut.result()
                if g:
                    graphs.append(g); ok += 1
                else:
                    fail += 1
                if i % 200 == 0:
                    print(f"    {i}/{total}  ok={ok}  fail={fail}")

    attrition = fail / total * 100 if total > 0 else 0
    print(f"  Done: {ok} graphs, {fail} failed ({attrition:.0f}% attrition)")
    return graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",     default=str(DATA / "MSR_data_cleaned.csv"))
    ap.add_argument("--subset",  type=int, default=None,
                    help="Process at most N vul=1 pairs (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("  Download: gdown 1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X -O data/MSR_data_cleaned.csv")
        sys.exit(1)

    DATA.mkdir(exist_ok=True)
    pairs, cleans = load_items(csv_path, args.subset, args.seed)

    print(f"\nSplitting {len(pairs):,} vuln pairs by CVE ...")
    pair_splits = _split_by_cve(pairs, args.seed)
    for name, ps in pair_splits.items():
        n_cves = len({p[4] for p in ps})
        print(f"  {name}: {len(ps):,} pairs from {n_cves:,} CVEs")

    print(f"\nSplitting {len(cleans):,} clean functions randomly ...")
    clean_splits = _split_random(cleans, args.seed)
    for name, cs in clean_splits.items():
        print(f"  {name}: {len(cs):,}")

    counter = 0
    for split_name in ["train", "valid", "test"]:
        print(f"\n── {split_name} ────────────────────────────────────────────")

        work: list[tuple] = []
        for (orig_idx, before, after, cwe, _cve) in pair_splits[split_name]:
            work.append((before, 1, cwe, counter));     counter += 1
            work.append((after,  0, cwe, counter));     counter += 1
        for (orig_idx, src, cwe) in clean_splits[split_name]:
            work.append((src, 0, cwe, counter));        counter += 1

        n_pos = sum(1 for w in work if w[1] == 1)
        n_neg = len(work) - n_pos
        print(f"  Work items: {len(work):,}  (pos={n_pos:,}, neg={n_neg:,})")

        graphs = compile_split(work, args.workers)
        n_g_pos = sum(1 for g in graphs if g["y"] == 1)
        n_g_neg = len(graphs) - n_g_pos
        print(f"  Graphs:     {len(graphs):,}  (pos={n_g_pos:,}, neg={n_g_neg:,})")

        out = DATA / f"bigvul_cls_{split_name}_instr_v2_graphs.pkl"
        with open(out, "wb") as f:
            pickle.dump(graphs, f)
        print(f"  Saved -> {out}")

    print("\nDone. Run train_bigvul_cls.py next.\n")


if __name__ == "__main__":
    main()
