#!/usr/bin/env python3
"""
preprocess_5.py — Instruction-level LLVM IR graph builder for Devign.

Same clang compilation pipeline as preprocess.py but builds one node per IR
instruction rather than one per basic block. Requires llvmlite.

Node layout:
  gid=0          : virtual "function entry" node (all function arguments map here)
  gid=1..N-1     : one node per IR instruction, in block/instruction order

Node features (32 total):
  feat[0]       : is_entry (1 only for the virtual entry node)
  feat[1..30]   : opcode one-hot over 30 common LLVM opcodes (0 for unknown)
  feat[31]      : is_dangerous_call (1 if call targets a known-unsafe API)

Edge types:
  0 = CFG  sequential within-block (instr_i → instr_{i+1}) plus
           branch-target cross-block (terminator → first instr of successor)
  1 = DFG  SSA use-def chains including phi nodes and function arguments

Output: data/{train,valid,test}_instr_graphs.pkl (separate from block-level pkl)

Usage:
    python preprocess_5.py                   # full dataset
    python preprocess_5.py --subset 1000     # quick sanity check
    python preprocess_5.py --workers 8       # parallel compilation (default: 4)
    python preprocess_5.py --seed 0          # different random subset
"""

import argparse
import json
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import llvmlite.binding as llvm
import numpy as np

# Reuse the full compilation pipeline (clang, preamble, stub injector) from
# preprocess.py without duplicating it. Import at module level so forked
# worker processes inherit the resolved PREAMBLE and _PROJECT_CFLAGS.
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from preprocess import compile_to_ir, DATA, _DANGEROUS_APIS  # noqa: E402

N_INSTR_FEATURES = 32

# 30 common LLVM -O0 opcodes; one-hot encoded as feat[1..30].
# Instructions with opcodes not in this list get all-zero opcode flags —
# the model still receives graph-structure signal for those instructions.
_INSTR_OPCODES = [
    "alloca", "load", "store", "getelementptr",   # memory
    "add", "sub", "mul", "sdiv", "udiv",          # integer arithmetic
    "fadd", "fsub", "fmul",                        # float arithmetic
    "shl", "lshr", "ashr", "and", "or", "xor",   # bitwise
    "trunc", "zext", "sext", "bitcast",           # type conversion
    "br", "ret", "unreachable", "switch",         # control flow
    "icmp", "call", "phi", "select",              # other
]
assert len(_INSTR_OPCODES) == 30, "opcode list must have exactly 30 entries"
_OPC_IDX      = {opc: i + 1 for i, opc in enumerate(_INSTR_OPCODES)}  # 1-based
_DANGEROUS_SET = set(_DANGEROUS_APIS)


# ---------------------------------------------------------------------------
# IR → instruction-level graph
# ---------------------------------------------------------------------------

def ir_to_instr_graph(ir_text: str) -> dict | None:
    """
    Parse LLVM IR and return an instruction-level graph dict:
      x          : (N, 32) float32   node features
      edge_index : (2, E) int64      CFG + DFG edges
      edge_type  : (E,) int64        0=CFG, 1=DFG

    Returns None on parse failure or degenerate IR.
    """
    try:
        mod = llvm.parse_assembly(ir_text)
    except Exception:
        return None

    # The user's function is always the LAST non-declaration.
    # The preamble (project headers) defines earlier inline functions.
    user_func = None
    for func in mod.functions:
        if not func.is_declaration:
            user_func = func
    if user_func is None:
        return None

    func   = user_func
    blocks = list(func.blocks)
    if not blocks:
        return None

    # ── Pre-pass: assign gids, build instr_id and block_first ──────────────
    # gid=0 is the virtual entry node; real instructions start at gid=1.
    instr_id:    dict[str, int] = {}
    block_first: dict[str, int] = {}

    # Function arguments → all map to the virtual entry node (gid=0).
    # Without this, uses of %arg in the first block have no def in instr_id
    # and their DFG edges would be silently dropped.
    for arg in func.arguments:
        if arg.name:
            instr_id[arg.name] = 0

    gid = 1
    for block in blocks:
        block_first[block.name] = gid
        for instr in block.instructions:
            if instr.name:           # unnamed instrs (br, ret, store…) stay out
                instr_id[instr.name] = gid
            gid += 1

    if gid < 2:
        return None   # no real instructions

    # ── Node features ───────────────────────────────────────────────────────
    # Virtual entry node: is_entry=1, everything else=0
    x: list[list[float]] = [[1.0] + [0.0] * (N_INSTR_FEATURES - 1)]

    for block in blocks:
        for instr in block.instructions:
            feat = [0.0] * N_INSTR_FEATURES
            # feat[0] stays 0 (is_entry = 0 for real instructions)
            opc_i = _OPC_IDX.get(instr.opcode, -1)
            if opc_i >= 0:
                feat[opc_i] = 1.0
            # feat[31]: is_dangerous_call — set if callee matches our API list
            if instr.opcode == "call":
                for op in instr.operands:
                    if op.name in _DANGEROUS_SET:
                        feat[31] = 1.0
                        break
            x.append(feat)

    # ── Edges ───────────────────────────────────────────────────────────────
    src_list:  list[int] = []
    dst_list:  list[int] = []
    type_list: list[int] = []
    seen_cfg: set[tuple[int, int]] = set()
    seen_dfg: set[tuple[int, int]] = set()

    def add_cfg(s: int, d: int) -> None:
        e = (s, d)
        if e not in seen_cfg:
            seen_cfg.add(e)
            src_list.append(s); dst_list.append(d); type_list.append(0)

    def add_dfg(s: int, d: int) -> None:
        e = (s, d)
        if e not in seen_dfg:
            seen_dfg.add(e)
            src_list.append(s); dst_list.append(d); type_list.append(1)

    gid = 1
    for block in blocks:
        instrs = list(block.instructions)
        n      = len(instrs)
        for i, instr in enumerate(instrs):
            cur = gid + i

            # Sequential CFG edge within block
            if i < n - 1:
                add_cfg(cur, cur + 1)

            # Branch-target CFG edges (from terminator instruction only).
            # llvmlite exposes branch targets as label-typed operands;
            # op.type == 'label' distinguishes them from value operands.
            if i == n - 1 and instr.opcode in ("br", "switch"):
                for op in instr.operands:
                    if str(op.type) == "label":
                        tgt = block_first.get(op.name)
                        if tgt is not None:
                            add_cfg(cur, tgt)

            # DFG edges: SSA use-def.
            # Skip unnamed operands (op.name == '') — these are constants.
            # Skip label-typed operands — these are branch targets, not values.
            # Phi operands handled automatically: llvmlite exposes only the
            # value operands of phi nodes (not the incoming-block labels).
            for op in instr.operands:
                if op.name and str(op.type) != "label":
                    def_gid = instr_id.get(op.name)
                    if def_gid is not None:
                        add_dfg(def_gid, cur)

        gid += n

    x_arr      = np.array(x, dtype=np.float32)
    edge_index = (np.array([src_list, dst_list], dtype=np.int64)
                  if src_list else np.zeros((2, 0), dtype=np.int64))
    edge_type  = (np.array(type_list, dtype=np.int64)
                  if type_list else np.zeros(0, dtype=np.int64))

    return {"x": x_arr, "edge_index": edge_index, "edge_type": edge_type}


# ---------------------------------------------------------------------------
# Per-item worker (called in parallel)
# ---------------------------------------------------------------------------

def process_item(item: dict) -> dict | None:
    ir = compile_to_ir(item["func"])
    if ir is None:
        return None
    g = ir_to_instr_graph(ir)
    if g is None:
        return None
    g["y"]   = int(item["target"])
    g["idx"] = item.get("idx", 0)
    return g


# ---------------------------------------------------------------------------
# Split processing
# ---------------------------------------------------------------------------

def process_split(jsonl_path: Path, subset: int | None, workers: int,
                  seed: int = 42, name: str = "") -> list[dict]:
    with open(jsonl_path) as f:
        items = [json.loads(l) for l in f]
    for i, item in enumerate(items):
        item["idx"] = i

    if subset:
        rng   = random.Random(seed)
        vuln  = [x for x in items if x["target"] == 1]
        fixed = [x for x in items if x["target"] == 0]
        half  = subset // 2
        items = (rng.sample(vuln,  min(half, len(vuln))) +
                 rng.sample(fixed, min(half, len(fixed))))
        rng.shuffle(items)

    label = name or jsonl_path.stem
    print(f"\n── {label} ───────────────────────────────────────────────")
    print(f"  Processing {len(items)} functions with {workers} workers ...")

    graphs: list[dict] = []
    ok = failed = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_item, item): item for item in items}
        for fut in as_completed(futs):
            result = fut.result()
            if result is not None:
                graphs.append(result)
                ok += 1
            else:
                failed += 1
            done = ok + failed
            if done % 500 == 0:
                print(f"    {done}/{len(items)}  ok={ok}  failed={failed}")

    pct = int(100 * failed / len(items)) if items else 0
    print(f"  Done: {ok} graphs built, {failed} failed ({pct}% attrition)")
    return graphs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset",  type=int, default=None,
                    help="Balanced random sample per split (quick test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    for split in ["train", "valid", "test"]:
        if not (DATA / f"{split}.jsonl").exists():
            print(f"Missing data/{split}.jsonl — run preprocess.py first.")
            sys.exit(1)

    for split in ["train", "valid", "test"]:
        graphs = process_split(
            DATA / f"{split}.jsonl",
            subset=args.subset,
            workers=args.workers,
            seed=args.seed,
            name=split,
        )
        out = DATA / f"{split}_instr_graphs.pkl"
        with open(out, "wb") as f:
            pickle.dump(graphs, f)
        print(f"  Saved {len(graphs)} graphs → {out}")


if __name__ == "__main__":
    main()
