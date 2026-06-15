#!/usr/bin/env python3
"""
preprocess_slice.py — Backward program-slice graphs from Devign LLVM IR.

§11 experiment: instead of classifying the full 400-node instruction graph,
extract the backward data-flow slice from dangerous sink call sites (strcpy,
memcpy, malloc, free, etc.) and GEP-with-variable-index nodes.

A 400-node graph where 3-5 nodes carry vulnerability signal becomes a 15-50
node slice where every node is on the dependency path to a dangerous operation.
Signal concentration goes from ~1% to ~50-80%.

Algorithm:
  1. Build full instruction-level graph (identical to preprocess_instr.py)
     — additionally track mock node names during Pass 3.
  2. Identify dangerous sink nodes:
       a. call instructions (opcode 63) whose function operand is in DANGEROUS_SINKS
       b. GEP instructions (opcode 29) with at least one non-constant DFG predecessor
  3. BFS backward through DFG edges (full closure, no depth limit) from each sink.
  4. Re-add a virtual context node; rebuild global context edges.
  5. Fallback: if no dangerous sinks found, keep the full graph (don't discard).

Output: data/{train,valid,test}_slice_graphs.pkl
  Same format as _instr_graphs.pkl — drop-in for train_slice.py.

Usage:
    python preprocess_slice.py --subset 200 --workers 1   # smoke test
    python preprocess_slice.py                             # full Devign
    python preprocess_slice.py --workers 8
"""

import argparse
import ctypes
import json
import pickle
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import llvmlite.binding as llvm

HERE = Path(__file__).parent
DATA = HERE / "data"

sys.path.insert(0, str(HERE))
from preprocess import compile_to_ir, download_devign

# ---------------------------------------------------------------------------
# Opcode vocabulary (identical to preprocess_instr.py — 110 entries)
# ---------------------------------------------------------------------------

OPCODE_VOCAB: dict[str, int] = {
    "add": 2,  "sub": 3,  "mul": 4,  "udiv": 5,  "sdiv": 6,
    "urem": 7, "srem": 8, "shl": 9,  "lshr": 10, "ashr": 11,
    "and": 12, "or": 13,  "xor": 14,
    "fadd": 15, "fsub": 16, "fmul": 17, "fdiv": 18, "frem": 19,
    "fneg": 20, "extractelement": 21, "insertelement": 22, "shufflevector": 23,
    "alloca": 26, "load": 27, "store": 28, "getelementptr": 29,
    "fence": 30, "cmpxchg": 31, "atomicrmw": 32,
    "br": 36, "switch": 37, "ret": 38, "invoke": 39,
    "resume": 40, "unreachable": 41, "indirectbr": 42, "callbr": 43,
    "icmp": 46, "fcmp": 47,
    "trunc": 48, "zext": 49, "sext": 50, "fptrunc": 51, "fpext": 52,
    "fptoui": 53, "fptosi": 54, "uitofp": 55, "sitofp": 56,
    "ptrtoint": 57, "inttoptr": 58, "bitcast": 59, "addrspacecast": 60,
    "phi": 61, "select": 62, "call": 63, "extractvalue": 64,
    "insertvalue": 65, "va_arg": 66, "landingpad": 67, "freeze": 68,
}
VOCAB_SIZE = 110

IDX_CONTEXT   = 0
IDX_ARGUMENT  = 1
IDX_MOCK      = 75
IDX_CONST_INT = 76
IDX_CONST_FP  = 77
IDX_UNDEF     = 78
IDX_UNKNOWN   = 79

_ICMP_PRED_RE = re.compile(r'\bicmp\s+(\w+)\b')
_FCMP_PRED_RE = re.compile(r'\bfcmp\s+(\w+)\b')

_ICMP_PRED_IDS: dict[str, int] = {
    "eq": 80,  "ne": 81,
    "slt": 82, "sle": 83, "sgt": 84, "sge": 85,
    "ult": 86, "ule": 87, "ugt": 88, "uge": 89,
}
_FCMP_PRED_IDS: dict[str, int] = {
    "false": 90, "oeq": 91, "ogt": 92, "oge": 93,
    "olt":  94,  "ole": 95, "one": 96, "ord": 97,
    "uno":  98,  "ueq": 99, "ugt": 100, "uge": 101,
    "ult":  102, "ule": 103, "une": 104, "true": 105,
}

VK_ARGUMENT     = 0
VK_BASIC_BLOCK  = 1
VK_FUNCTION     = 5
VK_GLOBAL_VAR   = 8
VK_UNDEF        = 14
VK_CONSTANT_INT = 18
VK_CONSTANT_FP  = 19
VK_INSTRUCTION  = 24
VK_POISON       = 25


def _instr_node_id(instr) -> int:
    op = instr.opcode
    if op == "icmp":
        m = _ICMP_PRED_RE.search(str(instr))
        if m:
            return _ICMP_PRED_IDS.get(m.group(1), IDX_UNKNOWN)
        return 46
    if op == "fcmp":
        m = _FCMP_PRED_RE.search(str(instr))
        if m:
            return _FCMP_PRED_IDS.get(m.group(1), IDX_UNKNOWN)
        return 47
    return OPCODE_VOCAB.get(op, IDX_UNKNOWN)


def _ptr_id(v) -> int:
    return ctypes.cast(v._ptr, ctypes.c_void_p).value


# ---------------------------------------------------------------------------
# Dangerous sink patterns
# ---------------------------------------------------------------------------

DANGEROUS_SINKS = frozenset({
    # Buffer copy / overflow (CWE-119, CWE-787, CWE-125)
    "strcpy", "strncpy", "strcat", "strncat",
    "memcpy", "memmove", "memset", "bcopy",
    "sprintf", "snprintf", "vsprintf", "vsnprintf",
    "gets", "fgets", "scanf", "sscanf", "fscanf",
    "read", "recv", "recvfrom", "pread",
    # Memory management (CWE-416 use-after-free, CWE-476 null-deref)
    "malloc", "calloc", "realloc", "free", "xmalloc", "xrealloc",
    # Format string (CWE-134)
    "printf", "fprintf", "syslog", "err", "warn",
})

_SINK_SUFFIXES = tuple(DANGEROUS_SINKS)


def _is_dangerous(name: str) -> bool:
    """Match 'strcpy', '__GI_strcpy', '__wrap_malloc', 'g_malloc', etc."""
    name = name.lstrip("@")
    if name in DANGEROUS_SINKS:
        return True
    # handle __GI_strcpy, __wrap_free, __libc_malloc, etc.
    for s in _SINK_SUFFIXES:
        if name.endswith(s) or name.endswith("_" + s):
            return True
    return False


# ---------------------------------------------------------------------------
# Backward slice extractor
# ---------------------------------------------------------------------------

_CONSTANT_IDS = frozenset({IDX_CONST_INT, IDX_CONST_FP, IDX_UNDEF, IDX_CONTEXT})


def _extract_slice(x: np.ndarray, edge_index: np.ndarray,
                   edge_type: np.ndarray, mock_names: dict) -> dict | None:
    """
    Backward DFG slice from dangerous sink nodes.

    Sinks:
      1. call instructions (opcode 63) whose function operand is a dangerous mock
      2. GEP instructions (opcode 29) with a non-constant DFG predecessor

    BFS backward through DFG edges (full closure). Rebuilds a virtual context
    node connecting all slice nodes with global context edges (type 2).

    Returns None if no sinks found (caller uses full graph as fallback).
    """
    E = edge_index.shape[1] if edge_index.ndim == 2 and edge_index.shape[1] > 0 else 0

    # Build forward and reverse DFG adjacency
    fwd_dfg: dict[int, list] = defaultdict(list)
    rev_dfg: dict[int, list] = defaultdict(list)
    for i in range(E):
        if int(edge_type[i]) == 1:
            s, d = int(edge_index[0, i]), int(edge_index[1, i])
            fwd_dfg[s].append(d)
            rev_dfg[d].append(s)

    # Sink type 1: dangerous call sites
    dangerous_mocks = {nid for nid, nm in mock_names.items() if _is_dangerous(nm)}
    sink_ids: set[int] = set()
    for mid in dangerous_mocks:
        for consumer in fwd_dfg[mid]:
            if int(x[consumer, 0]) == 63:   # call opcode
                sink_ids.add(consumer)

    # Sink type 2: GEP with non-constant index (potential OOB access)
    for i in range(E):
        if int(edge_type[i]) == 1:
            s, d = int(edge_index[0, i]), int(edge_index[1, i])
            if int(x[d, 0]) == 29 and int(x[s, 0]) not in _CONSTANT_IDS:
                sink_ids.add(d)

    if not sink_ids:
        return None

    # BFS backward through DFG edges — full closure
    visited: set[int] = set(sink_ids)
    frontier = list(sink_ids)
    while frontier:
        nxt = []
        for node in frontier:
            for pred in rev_dfg[node]:
                if pred not in visited and pred != 0:   # skip old context node
                    visited.add(pred)
                    nxt.append(pred)
        frontier = nxt

    # Re-index: new context node at position 0
    slice_nodes = sorted(visited)
    slice_size  = len(slice_nodes) + 1          # +1 for new context node
    old_to_new  = {old: new + 1 for new, old in enumerate(slice_nodes)}

    if slice_size < 2:
        return None

    # Build new_x
    new_x = np.zeros((slice_size, 1), dtype=np.int64)
    new_x[0, 0] = IDX_CONTEXT
    for new_id, old_id in enumerate(slice_nodes, start=1):
        new_x[new_id, 0] = int(x[old_id, 0])

    # Carry over CFG (0) and DFG (1) edges within the slice
    new_src, new_dst, new_et = [], [], []
    for i in range(E):
        et = int(edge_type[i])
        if et == 2:
            continue    # skip old global context edges; rebuilt below
        s, d = int(edge_index[0, i]), int(edge_index[1, i])
        if s in old_to_new and d in old_to_new:
            new_src.append(old_to_new[s])
            new_dst.append(old_to_new[d])
            new_et.append(et)

    # Bidirectional global context edges
    for new_id in range(1, slice_size):
        new_src.extend([new_id, 0])
        new_dst.extend([0, new_id])
        new_et.extend([2, 2])

    new_edge_index = (np.array([new_src, new_dst], dtype=np.int64)
                      if new_src else np.zeros((2, 0), dtype=np.int64))
    new_edge_type  = (np.array(new_et, dtype=np.int64)
                      if new_et  else np.zeros(0, dtype=np.int64))

    return {"x": new_x, "edge_index": new_edge_index, "edge_type": new_edge_type,
            "_sliced": True, "_n_sinks": len(sink_ids)}


# ---------------------------------------------------------------------------
# Graph builder — 5-pass algorithm + slice extraction
# ---------------------------------------------------------------------------

def ir_to_graph_slice(ir_text: str) -> dict | None:
    """
    Build instruction-level graph then extract backward slice from dangerous sinks.
    Returns None if IR parsing fails or result has < 2 nodes.
    The caller adds 'y' and 'idx' to the returned dict.
    """
    try:
        mod = llvm.parse_assembly(ir_text)
    except Exception:
        return None

    # Pass 0: find target function (last non-declaration)
    target_fn = None
    for fn in mod.functions:
        if not fn.is_declaration:
            target_fn = fn
    if target_fn is None:
        return None

    # Pass 1: allocate nodes
    node_opcodes: list[int] = []
    ptr_to_id:    dict[int, int] = {}
    node_counter  = 0

    node_opcodes.append(IDX_CONTEXT)
    node_counter = 1

    for arg in target_fn.arguments:
        ptr_to_id[_ptr_id(arg)] = node_counter
        node_opcodes.append(IDX_ARGUMENT)
        node_counter += 1

    block_first_instr: dict[int, int] = {}
    for block in target_fn.blocks:
        bpid = _ptr_id(block)
        first_in_block = True
        for instr in block.instructions:
            ipid = _ptr_id(instr)
            if first_in_block:
                block_first_instr[bpid] = node_counter
                first_in_block = False
            ptr_to_id[ipid] = node_counter
            node_opcodes.append(_instr_node_id(instr))
            node_counter += 1

    if node_counter < 2:
        return None

    edges_src:  list[int] = []
    edges_dst:  list[int] = []
    edges_type: list[int] = []

    # Pass 2: CFG edges (type 0)
    for block in target_fn.blocks:
        prev_id = None
        instrs  = list(block.instructions)
        for instr in instrs:
            cur_id = ptr_to_id[_ptr_id(instr)]
            if prev_id is not None:
                edges_src.append(prev_id)
                edges_dst.append(cur_id)
                edges_type.append(0)
            prev_id = cur_id
        if instrs:
            terminator = instrs[-1]
            term_id    = ptr_to_id[_ptr_id(terminator)]
            for op in terminator.operands:
                if op.value_kind == VK_BASIC_BLOCK:
                    succ_first = block_first_instr.get(_ptr_id(op))
                    if succ_first is not None:
                        edges_src.append(term_id)
                        edges_dst.append(succ_first)
                        edges_type.append(0)

    # Pass 3: DFG edges (type 1) — track mock names for sink identification
    constant_cache: dict[int, int] = {}
    mock_cache:     dict[str, int] = {}
    mock_names:     dict[int, str] = {}   # node_id -> function/global name

    for block in target_fn.blocks:
        for instr in block.instructions:
            dst_id = ptr_to_id[_ptr_id(instr)]
            for op in instr.operands:
                vk = op.value_kind

                if vk == VK_INSTRUCTION or vk == VK_ARGUMENT:
                    src_id = ptr_to_id.get(_ptr_id(op))
                    if src_id is not None:
                        edges_src.append(src_id)
                        edges_dst.append(dst_id)
                        edges_type.append(1)

                elif vk == VK_CONSTANT_INT:
                    opid = _ptr_id(op)
                    if opid not in constant_cache:
                        constant_cache[opid] = node_counter
                        node_opcodes.append(IDX_CONST_INT)
                        node_counter += 1
                    edges_src.append(constant_cache[opid])
                    edges_dst.append(dst_id)
                    edges_type.append(1)

                elif vk == VK_CONSTANT_FP:
                    opid = _ptr_id(op)
                    if opid not in constant_cache:
                        constant_cache[opid] = node_counter
                        node_opcodes.append(IDX_CONST_FP)
                        node_counter += 1
                    edges_src.append(constant_cache[opid])
                    edges_dst.append(dst_id)
                    edges_type.append(1)

                elif vk in (VK_GLOBAL_VAR, VK_FUNCTION):
                    name = op.name
                    if name not in mock_cache:
                        mock_cache[name]     = node_counter
                        mock_names[node_counter] = name   # ← track for sink detection
                        node_opcodes.append(IDX_MOCK)
                        node_counter += 1
                    edges_src.append(mock_cache[name])
                    edges_dst.append(dst_id)
                    edges_type.append(1)

                elif vk in (VK_UNDEF, VK_POISON):
                    opid = _ptr_id(op)
                    if opid not in constant_cache:
                        constant_cache[opid] = node_counter
                        node_opcodes.append(IDX_UNDEF)
                        node_counter += 1
                    edges_src.append(constant_cache[opid])
                    edges_dst.append(dst_id)
                    edges_type.append(1)

    # Pass 4: global context edges (type 2) — bidirectional
    for i in range(1, node_counter):
        edges_src.extend([i, 0])
        edges_dst.extend([0, i])
        edges_type.extend([2, 2])

    x          = np.array(node_opcodes, dtype=np.int64).reshape(-1, 1)
    edge_index = (np.array([edges_src, edges_dst], dtype=np.int64)
                  if edges_src else np.zeros((2, 0), dtype=np.int64))
    edge_type  = (np.array(edges_type, dtype=np.int64)
                  if edges_type else np.zeros(0, dtype=np.int64))

    # Extract backward slice from dangerous sinks
    g = _extract_slice(x, edge_index, edge_type, mock_names)
    if g is None:
        # No dangerous sinks — fall back to full instruction graph
        g = {"x": x, "edge_index": edge_index, "edge_type": edge_type,
             "_sliced": False, "_n_sinks": 0}

    return g


# ---------------------------------------------------------------------------
# Per-item processing (called in parallel workers)
# ---------------------------------------------------------------------------

def process_item_slice(item: dict) -> dict | None:
    ir = compile_to_ir(item["func"])
    if ir is None:
        return None
    g = ir_to_graph_slice(ir)
    if g is None:
        return None
    g["y"]   = int(item["target"])
    g["idx"] = item.get("idx", 0)
    return g


def process_split_slice(jsonl_path: Path, subset: int | None,
                        workers: int, seed: int = 42) -> list[dict]:
    with open(jsonl_path) as f:
        items = [json.loads(l) for l in f]

    rng = random.Random(seed)
    if subset:
        vuln  = [x for x in items if x["target"] == 1]
        fixed = [x for x in items if x["target"] == 0]
        rng.shuffle(vuln); rng.shuffle(fixed)
        items = vuln[:subset // 2] + fixed[:subset // 2]
    else:
        rng.shuffle(items)

    graphs, ok, fail = [], 0, 0
    total = len(items)
    print(f"  Processing {total} functions with {workers} workers ...")

    if workers == 1:
        for i, item in enumerate(items, 1):
            g = process_item_slice(item)
            if g:
                graphs.append(g)
                ok += 1
            else:
                fail += 1
            if i % 500 == 0:
                print(f"    {i}/{total}  ok={ok}  failed={fail}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(process_item_slice, it): it for it in items}
            for i, fut in enumerate(as_completed(futs), 1):
                g = fut.result()
                if g:
                    graphs.append(g)
                    ok += 1
                else:
                    fail += 1
                if i % 500 == 0:
                    print(f"    {i}/{total}  ok={ok}  failed={fail}")

    attrition = fail / total * 100 if total > 0 else 0
    print(f"  Done: {ok} graphs built, {fail} failed ({attrition:.0f}% attrition)")

    # Slice statistics
    node_counts = [g["x"].shape[0] for g in graphs]
    n_sliced    = sum(1 for g in graphs if g.get("_sliced", False))
    n_fallback  = ok - n_sliced
    if node_counts:
        print(f"  Slice stats: mean={np.mean(node_counts):.0f} nodes  "
              f"median={int(np.median(node_counts))}  max={max(node_counts)}")
        print(f"  Sliced: {n_sliced}/{ok} ({100*n_sliced/ok:.0f}%)  "
              f"Fallback (no sinks): {n_fallback}/{ok} ({100*n_fallback/ok:.0f}%)")

    # Strip internal fields before saving
    for g in graphs:
        g.pop("_sliced", None)
        g.pop("_n_sinks", None)

    return graphs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset",        type=int,  default=None)
    ap.add_argument("--workers",       type=int,  default=4)
    ap.add_argument("--seed",          type=int,  default=42)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        missing = any(not (DATA / f"{s}.jsonl").exists()
                      for s in ["train", "valid", "test"])
        if missing:
            print("\n-- Download --------------------------------------------------")
            download_devign()
        else:
            print("  data/*.jsonl present, skipping download.")

    for split in ["train", "valid", "test"]:
        src = DATA / f"{split}.jsonl"
        dst = DATA / f"{split}_slice_graphs.pkl"
        if not src.exists():
            print(f"Missing {src} -- run preprocess.py or drop --skip-download.")
            sys.exit(1)
        print(f"\n-- {split} ---------------------------------------------------")
        graphs = process_split_slice(src, subset=args.subset,
                                     workers=args.workers, seed=args.seed)
        with open(dst, "wb") as f:
            pickle.dump(graphs, f)
        print(f"  Saved {len(graphs)} graphs -> {dst}")

    print("\nDone. Run train_slice.py next.\n")


if __name__ == "__main__":
    main()
