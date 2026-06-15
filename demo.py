#!/usr/bin/env python3
"""
Structural LLVM IR embedding demo.

Extracts opcode-frequency histograms from compiled .ll files, then measures
pairwise cosine distances. The hypothesis: vulnerable and fixed versions of
the same code are structurally distinct in IR — missing guards, missing
branches, missing stores — and that difference should appear as larger
cross-class distances compared to within-class distances.

Usage:
    python3 demo.py [ir_dir]

ir_dir should contain <name>_vuln.ll and <name>_fixed.ll pairs.
Defaults to ./ir (produced by run.sh).
"""

import sys
import re
import math
from pathlib import Path
from collections import Counter

# Matches the opcode token from an LLVM IR instruction line.
# Handles both assigned form:  %x = icmp slt i32 ...
# and bare form:                br label %foo
_OPCODE_RE = re.compile(r"^\s+(?:%[\w.$\"]+ = )?(\w+)", re.MULTILINE)

# LLVM IR keywords that are not opcodes (types, attributes, directives).
_NON_OPCODES = {
    "define", "declare", "target", "source", "attributes", "module",
    "type", "global", "constant", "internal", "external", "private",
    "unnamed_addr", "local_unnamed_addr", "dso_local", "nounwind",
    "readnone", "speculatable", "willreturn", "nocallback", "nofree",
    "nosync", "nounwind", "i1", "i8", "i16", "i32", "i64", "i128",
    "float", "double", "void", "label", "align", "to", "from",
    "null", "true", "false", "zeroinitializer", "undef", "poison",
}


def extract_features(ll_path: Path) -> dict[str, float]:
    """Return a normalized opcode-frequency histogram for one .ll file."""
    text = ll_path.read_text(errors="replace")
    counts: Counter = Counter()
    for m in _OPCODE_RE.finditer(text):
        op = m.group(1)
        if op not in _NON_OPCODES and not op[0].isdigit():
            counts[op] += 1
    total = sum(counts.values()) or 1
    return {op: count / total for op, count in counts.items()}


def cosine_distance(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 1.0
    return 1.0 - dot / (mag_a * mag_b)


def main() -> None:
    ir_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ir")

    vuln_files = sorted(ir_dir.glob("*_vuln.ll"))
    if not vuln_files:
        print(f"No *_vuln.ll files found in {ir_dir}/")
        sys.exit(1)

    pairs: list[tuple[str, Path, Path]] = []
    for vf in vuln_files:
        name = vf.stem.replace("_vuln", "")
        ff = ir_dir / f"{name}_fixed.ll"
        if not ff.exists():
            print(f"  warning: no fixed counterpart for {name}, skipping")
            continue
        pairs.append((name, vf, ff))

    print(f"Loaded {len(pairs)} vulnerable/fixed pairs from {ir_dir}/\n")

    # Feature extraction
    features: dict[str, dict[str, float]] = {}
    for name, vf, ff in pairs:
        features[f"{name}_vuln"]  = extract_features(vf)
        features[f"{name}_fixed"] = extract_features(ff)

    # Per-pair: show which opcodes changed most
    print("=" * 60)
    print("TOP STRUCTURAL CHANGES PER PAIR  (fixed − vulnerable)")
    print("=" * 60)
    for name, _, _ in pairs:
        v = features[f"{name}_vuln"]
        f = features[f"{name}_fixed"]
        all_ops = set(v) | set(f)
        deltas = sorted(
            ((op, f.get(op, 0.0) - v.get(op, 0.0)) for op in all_ops),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:6]
        print(f"\n  {name}")
        for op, delta in deltas:
            bar = ("▲" if delta > 0 else "▼") * min(int(abs(delta) * 80) + 1, 20)
            print(f"    {op:<14} {delta:+.4f}  {bar}")

    # Distance matrix
    labels = []
    for name, _, _ in pairs:
        labels += [f"{name}_vuln", f"{name}_fixed"]

    n = len(labels)
    dist = [[cosine_distance(features[labels[i]], features[labels[j]])
             for j in range(n)] for i in range(n)]

    short = [l.replace("_vuln", "_V").replace("_fixed", "_F")
             for l in labels]
    col_w = max(len(s) for s in short) + 1

    print(f"\n\n{'=' * 60}")
    print("PAIRWISE COSINE DISTANCE MATRIX")
    print("(0 = identical structure,  1 = maximally different)")
    print("=" * 60)
    print(" " * (col_w + 2), end="")
    for s in short:
        print(f"{s:>{col_w}}", end="")
    print()
    for i, row_label in enumerate(labels):
        short_row = short[i]
        print(f"{short_row:<{col_w}}  ", end="")
        for j in range(n):
            val = dist[i][j]
            # Highlight diagonal (0), same-name pair, and cross-class
            print(f"{val:{col_w}.3f}", end="")
        print()

    # Summary
    vuln_vuln, fixed_fixed, cross = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            la, lb = labels[i], labels[j]
            d = dist[i][j]
            if "_vuln" in la and "_vuln" in lb:
                vuln_vuln.append(d)
            elif "_fixed" in la and "_fixed" in lb:
                fixed_fixed.append(d)
            else:
                cross.append(d)

    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0

    avg_vv = avg(vuln_vuln)
    avg_ff = avg(fixed_fixed)
    avg_cr = avg(cross)

    print(f"\n\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  avg vuln  ↔ vuln   distance : {avg_vv:.4f}")
    print(f"  avg fixed ↔ fixed  distance : {avg_ff:.4f}")
    print(f"  avg vuln  ↔ fixed  distance : {avg_cr:.4f}")

    within = avg(vuln_vuln + fixed_fixed)
    print(f"\n  separation ratio (cross / within-class) : {avg_cr / within:.2f}x"
          if within > 0 else "")

    print()
    if avg_cr > avg_vv and avg_cr > avg_ff:
        print("  ✓ SIGNAL PRESENT")
        print("    Cross-class distances are larger than within-class distances.")
        print("    Opcode histograms alone carry a structural signal.")
    else:
        print("  ✗ SIGNAL WEAK OR ABSENT")
        print("    Cross-class distances do not clearly exceed within-class.")
        print("    Raw opcode histograms may be insufficient — try subgraph features.")
    print()


if __name__ == "__main__":
    main()
