#!/usr/bin/env python3
"""
preprocess_instr_v6.py — §17 instruction-level graphs: taint propagation.

Builds on §16 (preprocess_instr_v5.py) with two improvements:

1. Extended Pattern B — unchecked dangerous return values:
   §16 Pattern B flagged only ALLOC functions whose result was never null-checked.
   §17 extends this to FILEIO and NETWORK functions (open, read, recv, send, …)
   whose result is never used as operand of any icmp anywhere in the function.
   These functions return -1 on error; an unchecked return is a common source of
   use-of-negative-value bugs (e.g. using the return as a buffer length).

2. Taint propagation through DFG — the core improvement:
   §16 flags source nodes (the dangerous call itself) but leaves downstream
   instructions unmarked. The RGCN already propagates information through edges,
   but an explicit taint feature in x[:,2] gives it a direct signal rather than
   requiring the model to discover the provenance chain implicitly.

   After computing initial flags (Pattern A + B), a BFS forward through DFG
   edges propagates taint with a 0.5 decay per hop (max 3 hops):
     source node:  x[:,2] = 1.0
     1 hop away:   x[:,2] = 0.5
     2 hops away:  x[:,2] = 0.25
     3 hops away:  x[:,2] = 0.125

   Each node takes the maximum taint value from any path reaching it.
   The feature is now a continuous float in [0, 1] rather than binary.

   Expected coverage increase: graph coverage stays ~14–20% (same source
   patterns), but node density within flagged graphs increases substantially
   — a flagged malloc and its downstream dereferences all carry taint signal.

Based on §13 (v2): 3 relations (CFG/DFG/Global). Same architecture as §16.

Node feature shape: (N, 3) float32
  x[:,0] = opcode_id         (cast to long for nn.Embedding)
  x[:,1] = const_magnitude   (Perfograph encoding, 0.0 for non-constants)
  x[:,2] = taint_value       (0.0 = clean, 1.0 = flagged source, 0.5/0.25/0.125 = propagated)

Outputs: data/{train,valid,test}_instr_v6_graphs.pkl

Usage:
    python preprocess_instr_v6.py --subset 200 --skip-download
    python preprocess_instr_v6.py --skip-download
    python preprocess_instr_v6.py --workers 8 --skip-download
"""

import argparse
import ctypes
import json
import math
import pickle
import random
import re
import struct
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import llvmlite.binding as llvm

HERE = Path(__file__).parent
DATA = HERE / "data"

sys.path.insert(0, str(HERE))
from preprocess import compile_to_ir, download_devign

# ---------------------------------------------------------------------------
# Opcode vocabulary (identical to v2–v5)
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
VOCAB_SIZE = 111

IDX_CONTEXT   = 0
IDX_ARGUMENT  = 1
IDX_MOCK      = 75
IDX_CONST_INT = 76
IDX_CONST_FP  = 77
IDX_UNDEF     = 78
IDX_UNKNOWN   = 79

IDX_MOCK_ALLOC   = 106
IDX_MOCK_COPY    = 107
IDX_MOCK_STRING  = 108
IDX_MOCK_FILEIO  = 109
IDX_MOCK_NETWORK = 110

# ---------------------------------------------------------------------------
# Categorical call targets (identical to v2–v5)
# ---------------------------------------------------------------------------

_CALL_BUCKETS: dict[str, int] = {}
for _fn in ("malloc", "calloc", "realloc", "free",
            "kmalloc", "kzalloc", "kfree", "vmalloc", "vfree",
            "av_malloc", "av_mallocz", "av_realloc", "av_free", "av_freep",
            "g_malloc", "g_malloc0", "g_realloc", "g_free", "g_new"):
    _CALL_BUCKETS[_fn] = IDX_MOCK_ALLOC
for _fn in ("memcpy", "memmove", "memset", "bcopy", "bzero"):
    _CALL_BUCKETS[_fn] = IDX_MOCK_COPY
for _fn in ("strcpy", "strncpy", "strcat", "strncat",
            "sprintf", "snprintf", "vsprintf", "vsnprintf",
            "gets", "fgets", "scanf", "sscanf", "fscanf",
            "strlen", "strcmp", "strncmp"):
    _CALL_BUCKETS[_fn] = IDX_MOCK_STRING
for _fn in ("fopen", "fclose", "fread", "fwrite", "fseek",
            "open", "close", "read", "write", "pread", "pwrite"):
    _CALL_BUCKETS[_fn] = IDX_MOCK_FILEIO
for _fn in ("recv", "recvfrom", "recvmsg", "send", "sendto", "sendmsg",
            "accept", "connect", "socket", "bind", "listen"):
    _CALL_BUCKETS[_fn] = IDX_MOCK_NETWORK


def _call_bucket_id(name: str) -> int:
    base = name.lstrip("_").split("@")[0]
    return _CALL_BUCKETS.get(base, _CALL_BUCKETS.get(name, IDX_MOCK))


# ---------------------------------------------------------------------------
# Perfograph constant encoding (identical to v2–v5)
# ---------------------------------------------------------------------------

_CONST_INT_VAL_RE = re.compile(r'^i\d+\s+(-?\d+)$')
_CONST_FP_VAL_RE  = re.compile(r'^\S+\s+(.+)$')


def _const_magnitude(val: float) -> float:
    if val == 0.0 or not math.isfinite(val):
        return 0.0
    try:
        return math.copysign(math.log2(abs(val) + 1.0), val)
    except (ValueError, OverflowError):
        return 0.0


def _extract_const_int(op) -> float:
    try:
        m = _CONST_INT_VAL_RE.match(str(op).strip())
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def _extract_const_fp(op) -> float:
    try:
        m = _CONST_FP_VAL_RE.match(str(op).strip())
        if not m:
            return 0.0
        val_s = m.group(1).strip()
        if val_s.startswith("0x"):
            raw = int(val_s, 16)
            return struct.unpack(">d", raw.to_bytes(8, "big"))[0]
        return float(val_s)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------

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


VK_ARGUMENT     = 0
VK_BASIC_BLOCK  = 1
VK_FUNCTION     = 5
VK_GLOBAL_VAR   = 8
VK_UNDEF        = 14
VK_CONSTANT_INT = 18
VK_CONSTANT_FP  = 19
VK_INSTRUCTION  = 24
VK_POISON       = 25


def _ptr_id(v) -> int:
    return ctypes.cast(v._ptr, ctypes.c_void_p).value


# ---------------------------------------------------------------------------
# §17: Static analysis flags + taint propagation
# ---------------------------------------------------------------------------

def _get_callee_name(instr) -> str | None:
    for op in instr.operands:
        if op.value_kind in (VK_GLOBAL_VAR, VK_FUNCTION):
            return op.name
    return None


# §17: extend unchecked-return pattern to FILEIO and NETWORK (return -1 on error)
_UNCHECKED_RETURN_CATEGORIES = {IDX_MOCK_ALLOC, IDX_MOCK_FILEIO, IDX_MOCK_NETWORK}


def _detect_static_flags(target_fn, ptr_to_id: dict) -> dict[int, float]:
    """
    Intra-procedural static analysis. Returns {node_id: 1.0} for source nodes.

    Pattern A — dangerous call without guard (identical to §16):
      Call to STRING/COPY/FILEIO/NETWORK with no icmp in same or predecessor block.

    Pattern B — unchecked return value (extended from §16):
      §16: ALLOC result never null-checked.
      §17: ALLOC / FILEIO / NETWORK result never used as operand of any icmp.
      Covers open/read/recv returning -1 on error in addition to malloc returning NULL.
    """
    flags: dict[int, float] = {}

    # ---- Extended Pattern B: collect candidates and find any icmp uses ----
    checked_return_ptrs: set[int] = set()   # call results that appear in any icmp
    candidate_ptrs:      dict[int, int] = {}  # ptr_id → node_id for candidates

    for block in target_fn.blocks:
        for instr in block.instructions:
            if instr.opcode == "call":
                callee = _get_callee_name(instr)
                if callee and _call_bucket_id(callee) in _UNCHECKED_RETURN_CATEGORIES:
                    nid = ptr_to_id.get(_ptr_id(instr))
                    if nid is not None:
                        candidate_ptrs[_ptr_id(instr)] = nid

    if candidate_ptrs:
        for block in target_fn.blocks:
            for instr in block.instructions:
                if instr.opcode == "icmp":
                    for op in instr.operands:
                        if _ptr_id(op) in candidate_ptrs:
                            checked_return_ptrs.add(_ptr_id(op))

    for ptr, nid in candidate_ptrs.items():
        if ptr not in checked_return_ptrs:
            flags[nid] = 1.0

    # ---- Build CFG predecessor map for Pattern A ----
    block_has_icmp: dict[int, bool] = {}
    block_preds:    dict[int, list[int]] = {}

    for block in target_fn.blocks:
        bpid = _ptr_id(block)
        block_has_icmp[bpid] = any(i.opcode == "icmp" for i in block.instructions)
        block_preds[bpid] = []

    for block in target_fn.blocks:
        bpid  = _ptr_id(block)
        instrs = list(block.instructions)
        if instrs:
            for op in instrs[-1].operands:
                if op.value_kind == VK_BASIC_BLOCK:
                    succ_pid = _ptr_id(op)
                    if succ_pid in block_preds:
                        block_preds[succ_pid].append(bpid)

    # ---- Pattern A: dangerous call without preceding icmp guard ----
    dangerous_a = {IDX_MOCK_STRING, IDX_MOCK_COPY, IDX_MOCK_FILEIO, IDX_MOCK_NETWORK}

    for block in target_fn.blocks:
        bpid = _ptr_id(block)
        pred_has_icmp = any(block_has_icmp.get(p, False) for p in block_preds[bpid])
        icmp_seen = False
        for instr in block.instructions:
            if instr.opcode == "icmp":
                icmp_seen = True
                continue
            if instr.opcode != "call":
                continue
            callee = _get_callee_name(instr)
            if callee and _call_bucket_id(callee) in dangerous_a:
                if not icmp_seen and not pred_has_icmp:
                    nid = ptr_to_id.get(_ptr_id(instr))
                    if nid is not None:
                        flags[nid] = 1.0

    return flags


def _propagate_taint(
    flags:      dict[int, float],
    edges_src:  list[int],
    edges_dst:  list[int],
    edges_type: list[int],
    decay:      float = 0.5,
    max_hops:   int   = 3,
    min_val:    float = 0.05,
) -> dict[int, float]:
    """
    BFS forward through DFG edges (type=1) from each flagged source node.
    Each hop multiplies the taint value by `decay`. Stops when value < min_val
    or max_hops is reached. Each node takes the maximum taint from any path.
    Returns a new dict with initial flags plus all propagated values.
    """
    # Build forward DFG adjacency (type=1 edges only)
    dfg_fwd: dict[int, list[int]] = {}
    for s, d, t in zip(edges_src, edges_dst, edges_type):
        if t == 1:
            dfg_fwd.setdefault(s, []).append(d)

    result = dict(flags)

    for start, start_val in flags.items():
        queue: deque[tuple[int, float, int]] = deque()
        queue.append((start, start_val, 0))
        seen: set[int] = {start}

        while queue:
            node, val, hop = queue.popleft()
            if hop >= max_hops:
                continue
            next_val = val * decay
            if next_val < min_val:
                continue
            for neighbor in dfg_fwd.get(node, []):
                if neighbor not in seen:
                    seen.add(neighbor)
                    result[neighbor] = max(result.get(neighbor, 0.0), next_val)
                    queue.append((neighbor, next_val, hop + 1))

    return result


# ---------------------------------------------------------------------------
# Instruction-level graph extractor — 5-pass algorithm
# ---------------------------------------------------------------------------

def ir_to_graph_instr(ir_text: str) -> dict | None:
    try:
        mod = llvm.parse_assembly(ir_text)
    except Exception:
        return None

    target_fn = None
    for fn in mod.functions:
        if not fn.is_declaration:
            target_fn = fn
    if target_fn is None:
        return None

    # -- Pass 1: Allocate nodes -----------------------------------------------
    node_opcodes:    list[int]   = []
    node_magnitudes: list[float] = []
    node_taints:     list[float] = []  # §17: taint value, updated in Pass 5
    ptr_to_id: dict[int, int] = {}
    node_counter = 0

    node_opcodes.append(IDX_CONTEXT)
    node_magnitudes.append(0.0)
    node_taints.append(0.0)
    node_counter = 1

    for arg in target_fn.arguments:
        ptr_to_id[_ptr_id(arg)] = node_counter
        node_opcodes.append(IDX_ARGUMENT)
        node_magnitudes.append(0.0)
        node_taints.append(0.0)
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
            node_magnitudes.append(0.0)
            node_taints.append(0.0)
            node_counter += 1

    if node_counter < 2:
        return None

    edges_src:  list[int] = []
    edges_dst:  list[int] = []
    edges_type: list[int] = []

    # -- Pass 2: CFG edges (type 0) -------------------------------------------
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

    # -- Pass 3: DFG edges (type 1) -------------------------------------------
    constant_cache: dict[int, int] = {}
    mock_cache:     dict[str, int] = {}

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
                        node_magnitudes.append(_const_magnitude(_extract_const_int(op)))
                        node_taints.append(0.0)
                        node_counter += 1
                    edges_src.append(constant_cache[opid])
                    edges_dst.append(dst_id)
                    edges_type.append(1)

                elif vk == VK_CONSTANT_FP:
                    opid = _ptr_id(op)
                    if opid not in constant_cache:
                        constant_cache[opid] = node_counter
                        node_opcodes.append(IDX_CONST_FP)
                        node_magnitudes.append(_const_magnitude(_extract_const_fp(op)))
                        node_taints.append(0.0)
                        node_counter += 1
                    edges_src.append(constant_cache[opid])
                    edges_dst.append(dst_id)
                    edges_type.append(1)

                elif vk in (VK_GLOBAL_VAR, VK_FUNCTION):
                    name = op.name
                    if name not in mock_cache:
                        mock_cache[name] = node_counter
                        node_opcodes.append(_call_bucket_id(name))
                        node_magnitudes.append(0.0)
                        node_taints.append(0.0)
                        node_counter += 1
                    edges_src.append(mock_cache[name])
                    edges_dst.append(dst_id)
                    edges_type.append(1)

                elif vk in (VK_UNDEF, VK_POISON):
                    opid = _ptr_id(op)
                    if opid not in constant_cache:
                        constant_cache[opid] = node_counter
                        node_opcodes.append(IDX_UNDEF)
                        node_magnitudes.append(0.0)
                        node_taints.append(0.0)
                        node_counter += 1
                    edges_src.append(constant_cache[opid])
                    edges_dst.append(dst_id)
                    edges_type.append(1)

    # -- Pass 4: Global Context edges (type 2) --------------------------------
    for i in range(1, node_counter):
        edges_src.extend([i, 0])
        edges_dst.extend([0, i])
        edges_type.extend([2, 2])

    # -- Pass 5: Static flags + taint propagation (§17) -----------------------
    # Detect source nodes, then BFS forward through DFG with decay.
    source_flags = _detect_static_flags(target_fn, ptr_to_id)
    taint_map    = _propagate_taint(source_flags, edges_src, edges_dst, edges_type)
    for nid, val in taint_map.items():
        if nid < len(node_taints):
            node_taints[nid] = val

    # -------------------------------------------------------------------------
    x          = np.column_stack([
                     np.array(node_opcodes,    dtype=np.float32),
                     np.array(node_magnitudes, dtype=np.float32),
                     np.array(node_taints,     dtype=np.float32),
                 ])
    edge_index = (np.array([edges_src, edges_dst], dtype=np.int64)
                  if edges_src else np.zeros((2, 0), dtype=np.int64))
    edge_type  = (np.array(edges_type, dtype=np.int64)
                  if edges_type else np.zeros(0, dtype=np.int64))

    return {"x": x, "edge_index": edge_index, "edge_type": edge_type}


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------

def process_item_instr(item: dict) -> dict | None:
    ir = compile_to_ir(item["func"])
    if ir is None:
        return None
    g = ir_to_graph_instr(ir)
    if g is None:
        return None
    g["y"]   = int(item["target"])
    g["idx"] = item.get("idx", 0)
    return g


def process_split_instr(jsonl_path: Path, subset: int | None,
                         workers: int, seed: int = 42) -> list[dict]:
    with open(jsonl_path) as f:
        items = [json.loads(l) for l in f]

    rng = random.Random(seed)
    if subset:
        vuln  = [x for x in items if x["target"] == 1]
        fixed = [x for x in items if x["target"] == 0]
        rng.shuffle(vuln)
        rng.shuffle(fixed)
        items = vuln[:subset // 2] + fixed[:subset // 2]
    else:
        rng.shuffle(items)

    graphs, ok, fail = [], 0, 0
    total = len(items)
    print(f"  Processing {total} functions with {workers} workers ...")

    if workers == 1:
        for i, item in enumerate(items, 1):
            g = process_item_instr(item)
            if g:
                graphs.append(g)
                ok += 1
            else:
                fail += 1
            if i % 500 == 0:
                print(f"    {i}/{total}  ok={ok}  failed={fail}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(process_item_instr, it): it for it in items}
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
    return graphs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset",        type=int, default=None)
    ap.add_argument("--workers",       type=int, default=4)
    ap.add_argument("--seed",          type=int, default=42)
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
        dst = DATA / f"{split}_instr_v6_graphs.pkl"
        if not src.exists():
            print(f"Missing {src} -- run preprocess.py or drop --skip-download.")
            sys.exit(1)
        print(f"\n-- {split} ---------------------------------------------------")
        graphs = process_split_instr(src, subset=args.subset,
                                      workers=args.workers, seed=args.seed)
        with open(dst, "wb") as f:
            pickle.dump(graphs, f)
        print(f"  Saved {len(graphs)} graphs -> {dst}")

    print("\nDone. Run train_instr_v6.py next.\n")


if __name__ == "__main__":
    main()
