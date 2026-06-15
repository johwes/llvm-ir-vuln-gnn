#!/usr/bin/env python3
"""
Minimal CFG/DFG graph extraction from LLVM IR text.
No dependencies beyond Python stdlib — no ProGraML, no llvmlite.

Extracts per function:
  CFG nodes : basic blocks
  CFG edges : branch targets (control flow)
  DFG edges : cross-block def-use (data flow)

Usage:
    python3 graph_demo.py [ir_dir]
    python3 graph_demo.py ir/nullderef_vuln.ll ir/nullderef_fixed.ll
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

_BB_LABEL  = re.compile(r'^([\w.]+):')
_DEF       = re.compile(r'^\s+(%[\w.]+)\s*=')
_USE       = re.compile(r'%[\w.]+')
_BR_COND   = re.compile(r'br i1 .+?label %(\w+).+?label %(\w+)')
_BR_UNCOND = re.compile(r'br label %(\w+)')
_DEFINE    = re.compile(r'^define\b.*@(\w+)\s*\(')


def extract_graphs(ll_path: Path) -> dict:
    """Return {func_name: {blocks, cfg_edges, dfg_edges}} for every function in a .ll file."""
    text = ll_path.read_text(errors="replace")
    results = {}

    in_func     = False
    func_name   = None
    current_bb  = None
    bb_list     = []
    cfg_edges   = []
    defs        = {}             # varname -> bb where defined
    uses        = defaultdict(list)  # varname -> [bb, ...]

    for line in text.splitlines():
        m = _DEFINE.match(line)
        if m:
            in_func    = True
            func_name  = m.group(1)
            current_bb = "entry"
            bb_list    = ["entry"]
            cfg_edges  = []
            defs       = {}
            uses       = defaultdict(list)
            continue

        if not in_func:
            continue

        if line.strip() == "}":
            dfg_edges = [
                (defs[v], ub, v)
                for v, use_bbs in uses.items()
                if v in defs
                for ub in use_bbs
                if ub != defs[v]
            ]
            results[func_name] = {
                "blocks":    bb_list,
                "cfg_edges": cfg_edges,
                "dfg_edges": dfg_edges,
            }
            in_func = False
            continue

        m = _BB_LABEL.match(line)
        if m:
            current_bb = m.group(1)
            if current_bb not in bb_list:
                bb_list.append(current_bb)
            continue

        if current_bb is None:
            continue

        # Record def
        m = _DEF.match(line)
        defined_var = None
        if m:
            defined_var = m.group(1)
            if defined_var not in defs:
                defs[defined_var] = current_bb

        # Record uses (exclude the variable being defined on this line)
        for var in _USE.findall(line):
            if var != defined_var:
                uses[var].append(current_bb)

        # CFG edges
        m = _BR_COND.search(line)
        if m:
            cfg_edges.append((current_bb, m.group(1)))
            cfg_edges.append((current_bb, m.group(2)))
        else:
            m = _BR_UNCOND.search(line)
            if m:
                cfg_edges.append((current_bb, m.group(1)))

    return results


def main() -> None:
    args = sys.argv[1:]
    if args:
        paths = [Path(a) for a in args]
        if len(paths) == 1 and paths[0].is_dir():
            paths = sorted(paths[0].glob("*.ll"))
    else:
        paths = sorted(Path("ir").glob("*.ll"))

    if not paths:
        print("No .ll files found.")
        sys.exit(1)

    # Collect all graphs keyed by (stem, func_name)
    all_graphs: dict[str, dict] = {}
    for path in paths:
        for fname, g in extract_graphs(path).items():
            all_graphs[f"{path.stem}/{fname}"] = g

    # Per-file summary
    print(f"\n{'='*62}")
    print(f"{'FILE/FUNCTION':<35} {'BLOCKS':>7} {'CFG':>5} {'DFG':>5}")
    print(f"{'='*62}")
    for label, g in sorted(all_graphs.items()):
        print(f"  {label:<33} {len(g['blocks']):>7} {len(g['cfg_edges']):>5} {len(g['dfg_edges']):>5}")

    # Per-pair diff
    stems  = sorted({k.split("/")[0] for k in all_graphs})
    names  = sorted({s.replace("_vuln", "").replace("_fixed", "") for s in stems})

    pairs = [
        n for n in names
        if any(f"{n}_vuln" in k for k in all_graphs)
        and any(f"{n}_fixed" in k for k in all_graphs)
    ]

    if not pairs:
        return

    print(f"\n\n{'='*62}")
    print("PER-PAIR STRUCTURAL DIFF  (fixed − vulnerable)")
    print(f"{'='*62}")
    print(f"{'PAIR':<14} {'BLOCKS':>10} {'CFG edges':>10} {'DFG edges':>10}")
    print(f"{'-'*14} {'-'*10} {'-'*10} {'-'*10}")

    for name in pairs:
        vkey = next(k for k in all_graphs if f"{name}_vuln" in k)
        fkey = next(k for k in all_graphs if f"{name}_fixed" in k)
        v = all_graphs[vkey]
        f = all_graphs[fkey]

        db = len(f["blocks"])    - len(v["blocks"])
        dc = len(f["cfg_edges"]) - len(v["cfg_edges"])
        dd = len(f["dfg_edges"]) - len(v["dfg_edges"])

        def fmt(vuln, fixed, delta):
            return f"{vuln}→{fixed} ({delta:+d})"

        print(f"  {name:<12} "
              f"  {fmt(len(v['blocks']),    len(f['blocks']),    db):>10}"
              f"  {fmt(len(v['cfg_edges']), len(f['cfg_edges']), dc):>10}"
              f"  {fmt(len(v['dfg_edges']), len(f['dfg_edges']), dd):>10}")

    print()
    print("  BLOCKS    = CFG nodes (basic blocks per function)")
    print("  CFG edges = control-flow transitions between blocks")
    print("  DFG edges = cross-block def→use data dependencies")
    print()


if __name__ == "__main__":
    main()
