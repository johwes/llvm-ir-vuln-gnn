#!/usr/bin/env python3
"""
preprocess_instr_v3.py — §14 instruction-level graphs: VSDG memory ordering edges.

Adds one improvement over preprocess_instr_v2.py (§13):

Pass 5 — State dependence edges (type 3):
  For each pair of consecutive load/store instructions that operate on the same
  pointer operand, add a directed state edge (earlier → later, type=3). This
  captures memory operation ordering that CFG and DFG edges miss:
    - load → store on same ptr: read-modify-write sequence
    - store → load on same ptr: write-then-read (UAF pattern if ptr was freed)
    - store → store on same ptr: double-write / aliasing

  Edge direction is execution order (earlier instr → later instr). Only
  consecutive pairs on each pointer are connected (O(n) per pointer, not O(n²)).
  Cross-block ordering is approximated by iteration order over blocks and
  instructions (topological for reducible CFGs with a single entry point).

Edge type summary (4 relations total):
  0 = CFG  (sequential + inter-block control flow)
  1 = DFG  (SSA def-use chains)
  2 = Global (bidirectional to virtual context node 0)
  3 = State  (memory operation ordering: load/store on same pointer)

Outputs: data/{train,valid,test}_instr_v3_graphs.pkl
  Each graph dict: {"x": float32 (N,2), "edge_index": int64 (2,E),
                    "edge_type": int64 (E,), "y": int, "idx": int}

Usage:
    python preprocess_instr_v3.py --subset 200   # smoke test
    python preprocess_instr_v3.py                # full Devign dataset
    python preprocess_instr_v3.py --workers 8    # parallel compilation
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import llvmlite.binding as llvm

HERE = Path(__file__).parent
DATA = HERE / "data"

sys.path.insert(0, str(HERE))
from preprocess import compile_to_ir, download_devign

# ---------------------------------------------------------------------------
# Opcode vocabulary (identical to v2)
# ---------------------------------------------------------------------------

OPCODE_VOCAB: dict[str, int] = {
    # ALU
    "add": 2,  "sub": 3,  "mul": 4,  "udiv": 5,  "sdiv": 6,
    "urem": 7, "srem": 8, "shl": 9,  "lshr": 10, "ashr": 11,
    "and": 12, "or": 13,  "xor": 14,
    "fadd": 15, "fsub": 16, "fmul": 17, "fdiv": 18, "frem": 19,
    "fneg": 20, "extractelement": 21, "insertelement": 22, "shufflevector": 23,
    # Memory
    "alloca": 26, "load": 27, "store": 28, "getelementptr": 29,
    "fence": 30, "cmpxchg": 31, "atomicrmw": 32,
    # Control-flow
    "br": 36, "switch": 37, "ret": 38, "invoke": 39,
    "resume": 40, "unreachable": 41, "indirectbr": 42, "callbr": 43,
    # Comparison
    "icmp": 46, "fcmp": 47,
    # Cast
    "trunc": 48, "zext": 49, "sext": 50, "fptrunc": 51, "fpext": 52,
    "fptoui": 53, "fptosi": 54, "uitofp": 55, "sitofp": 56,
    "ptrtoint": 57, "inttoptr": 58, "bitcast": 59, "addrspacecast": 60,
    # Other
    "phi": 61, "select": 62, "call": 63, "extractvalue": 64,
    "insertvalue": 65, "va_arg": 66, "landingpad": 67, "freeze": 68,
}
VOCAB_SIZE = 111  # unchanged from v2

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
# Categorical call targets (identical to v2)
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
# Perfograph constant encoding (identical to v2)
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
# Instruction-level graph extractor — 6-pass algorithm
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
    ptr_to_id: dict[int, int] = {}
    node_counter = 0

    # Node 0: Virtual Context Node
    node_opcodes.append(IDX_CONTEXT)
    node_magnitudes.append(0.0)
    node_counter = 1

    for arg in target_fn.arguments:
        ptr_to_id[_ptr_id(arg)] = node_counter
        node_opcodes.append(IDX_ARGUMENT)
        node_magnitudes.append(0.0)
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
                        node_magnitudes.append(
                            _const_magnitude(_extract_const_int(op)))
                        node_counter += 1
                    edges_src.append(constant_cache[opid])
                    edges_dst.append(dst_id)
                    edges_type.append(1)

                elif vk == VK_CONSTANT_FP:
                    opid = _ptr_id(op)
                    if opid not in constant_cache:
                        constant_cache[opid] = node_counter
                        node_opcodes.append(IDX_CONST_FP)
                        node_magnitudes.append(
                            _const_magnitude(_extract_const_fp(op)))
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
                        node_counter += 1
                    edges_src.append(constant_cache[opid])
                    edges_dst.append(dst_id)
                    edges_type.append(1)

    # -- Pass 4: Global Context edges (type 2) --------------------------------
    for i in range(1, node_counter):
        edges_src.extend([i, 0])
        edges_dst.extend([0, i])
        edges_type.extend([2, 2])

    # -- Pass 5: State dependence edges (type 3) ——  §14 ---------------------
    # Track load and store instructions by the pointer they operate on.
    # Consecutive memory ops on the same pointer get a directed state edge
    # (earlier → later) encoding ordering constraints that DFG misses.
    #
    # Pointer operand positions in LLVM IR:
    #   load  T, T* %ptr        → operands[0] is the pointer
    #   store T %val, T* %ptr   → operands[1] is the pointer
    #
    # Iteration order (block order, then instruction order within block)
    # approximates execution order for reducible CFGs.
    memory_ops: dict[int, list[int]] = {}  # ptr_ptr_id -> [node_ids in order]

    for block in target_fn.blocks:
        for instr in block.instructions:
            op_name = instr.opcode
            ptr_op  = None
            if op_name == "load":
                ops = list(instr.operands)
                if ops:
                    ptr_op = ops[0]
            elif op_name == "store":
                ops = list(instr.operands)
                if len(ops) >= 2:
                    ptr_op = ops[1]

            if ptr_op is not None:
                pid = _ptr_id(ptr_op)
                nid = ptr_to_id.get(_ptr_id(instr))
                if nid is not None:
                    memory_ops.setdefault(pid, []).append(nid)

    for mem_nids in memory_ops.values():
        for i in range(len(mem_nids) - 1):
            edges_src.append(mem_nids[i])
            edges_dst.append(mem_nids[i + 1])
            edges_type.append(3)

    # -------------------------------------------------------------------------
    x          = np.column_stack([
                     np.array(node_opcodes,    dtype=np.float32),
                     np.array(node_magnitudes, dtype=np.float32),
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
    ap.add_argument("--subset",       type=int, default=None)
    ap.add_argument("--workers",      type=int, default=4)
    ap.add_argument("--seed",         type=int, default=42)
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
        dst = DATA / f"{split}_instr_v3_graphs.pkl"
        if not src.exists():
            print(f"Missing {src} -- run preprocess.py or drop --skip-download.")
            sys.exit(1)
        print(f"\n-- {split} ---------------------------------------------------")
        graphs = process_split_instr(src, subset=args.subset,
                                      workers=args.workers, seed=args.seed)
        with open(dst, "wb") as f:
            pickle.dump(graphs, f)
        print(f"  Saved {len(graphs)} graphs -> {dst}")

    print("\nDone. Run train_instr_v3.py next.\n")


if __name__ == "__main__":
    main()
