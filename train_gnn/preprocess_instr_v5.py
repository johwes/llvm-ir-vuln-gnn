#!/usr/bin/env python3
"""
preprocess_instr_v5.py — §16 instruction-level graphs: static analysis flags.

Adds one improvement over preprocess_instr_v2.py (§13):

Static analysis flag (x[:,2]):
  GNNs struggle to detect vulnerability patterns that are *absences* — a
  missing bounds check, a malloc result never compared to null, a dangerous
  string function call with no preceding guard. The graph has no node for
  the icmp that should be there but isn't, so there is no gradient signal.

  This pass runs lightweight static analysis over the CFG before building
  the graph, explicitly converting those absences into a presence signal:
  a binary flag (0.0 = clean, 1.0 = suspicious) stored as x[:,2].

  Two patterns are detected:

  Pattern A — dangerous call without preceding guard:
    A call to a STRING/COPY/FILEIO/NETWORK function (e.g. strcpy, memcpy,
    fread, recv) where neither the current basic block nor any direct
    predecessor block contains an icmp instruction. An icmp in a predecessor
    means "there was a conditional check before reaching this call." Its
    absence means the call is unconditional — a common vulnerability pattern.

  Pattern B — allocation result never null-checked:
    A call to an ALLOC function (malloc, calloc, kmalloc, …) whose SSA result
    is never compared to null anywhere in the function. Dereferencing an
    unchecked malloc result is a classic null-deref vulnerability class.

  Both patterns are intra-procedural and work with standard -O0 IR (no -g
  required). They are deliberately conservative: only flag when the evidence
  of a missing guard is unambiguous.

Based on §13 (v2): 3 relations (CFG/DFG/Global), Perfograph constants,
categorical call targets. Drops §14 state edges (no improvement) and §15
name embedding (no improvement).

Node feature shape: (N, 3) float32
  x[:,0] = opcode_id         (cast to long for nn.Embedding)
  x[:,1] = const_magnitude   (Perfograph encoding, 0.0 for non-constants)
  x[:,2] = static_flag       (0.0 = clean, 1.0 = suspicious)

Outputs: data/{train,valid,test}_instr_v5_graphs.pkl

Usage:
    python preprocess_instr_v5.py --subset 200 --skip-download
    python preprocess_instr_v5.py --skip-download
    python preprocess_instr_v5.py --workers 8 --skip-download
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
# Opcode vocabulary (identical to v2/v3/v4)
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
# Categorical call targets (identical to v2/v3/v4)
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
# Perfograph constant encoding (identical to v2/v3/v4)
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
# §16: Static analysis flag detection
# ---------------------------------------------------------------------------

def _get_callee_name(instr) -> str | None:
    """Return the name of the called function, or None for indirect calls."""
    for op in instr.operands:
        if op.value_kind in (VK_GLOBAL_VAR, VK_FUNCTION):
            return op.name
    return None


def _detect_static_flags(target_fn, ptr_to_id: dict) -> dict[int, float]:
    """
    Lightweight intra-procedural static analysis over the CFG.
    Returns {node_id: 1.0} for instruction nodes flagged as suspicious.

    Pattern A — dangerous call without guard:
      A call to a STRING/COPY/FILEIO/NETWORK function where no icmp appears
      before it in the same block, and no direct predecessor block contains
      an icmp. Checks direct predecessors (1 hop) to catch the common pattern
      of a guard block flowing into the call block.

    Pattern B — unchecked allocation:
      A call to an ALLOC function whose SSA result is never operand of an
      icmp-null comparison anywhere in the function.
    """
    flags: dict[int, float] = {}

    # ---- Pre-scan: collect alloc results, then find null checks ----
    alloc_result_ptrs: set[int] = set()
    for block in target_fn.blocks:
        for instr in block.instructions:
            if instr.opcode != "call":
                continue
            callee = _get_callee_name(instr)
            if callee and _call_bucket_id(callee) == IDX_MOCK_ALLOC:
                alloc_result_ptrs.add(_ptr_id(instr))

    null_checked_ptrs: set[int] = set()
    if alloc_result_ptrs:
        for block in target_fn.blocks:
            for instr in block.instructions:
                if instr.opcode == "icmp" and "null" in str(instr):
                    for op in instr.operands:
                        if _ptr_id(op) in alloc_result_ptrs:
                            null_checked_ptrs.add(_ptr_id(op))

    # ---- Pattern B: flag each unchecked alloc call ----
    for block in target_fn.blocks:
        for instr in block.instructions:
            if instr.opcode != "call":
                continue
            callee = _get_callee_name(instr)
            if callee and _call_bucket_id(callee) == IDX_MOCK_ALLOC:
                if _ptr_id(instr) not in null_checked_ptrs:
                    nid = ptr_to_id.get(_ptr_id(instr))
                    if nid is not None:
                        flags[nid] = 1.0

    # ---- Build CFG: per-block icmp presence and predecessor map ----
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
    dangerous = {IDX_MOCK_STRING, IDX_MOCK_COPY, IDX_MOCK_FILEIO, IDX_MOCK_NETWORK}

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
            if callee is None:
                continue
            if _call_bucket_id(callee) in dangerous:
                if not icmp_seen and not pred_has_icmp:
                    nid = ptr_to_id.get(_ptr_id(instr))
                    if nid is not None:
                        flags[nid] = 1.0

    return flags


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
    node_flags:      list[float] = []  # §16: static analysis flag
    ptr_to_id: dict[int, int] = {}
    node_counter = 0

    # Node 0: Virtual Context Node
    node_opcodes.append(IDX_CONTEXT)
    node_magnitudes.append(0.0)
    node_flags.append(0.0)
    node_counter = 1

    for arg in target_fn.arguments:
        ptr_to_id[_ptr_id(arg)] = node_counter
        node_opcodes.append(IDX_ARGUMENT)
        node_magnitudes.append(0.0)
        node_flags.append(0.0)
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
            node_flags.append(0.0)  # updated in Pass 5
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
                        node_flags.append(0.0)  # constants cannot be flagged
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
                        node_flags.append(0.0)
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
                        node_flags.append(0.0)
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
                        node_flags.append(0.0)
                        node_counter += 1
                    edges_src.append(constant_cache[opid])
                    edges_dst.append(dst_id)
                    edges_type.append(1)

    # -- Pass 4: Global Context edges (type 2) --------------------------------
    for i in range(1, node_counter):
        edges_src.extend([i, 0])
        edges_dst.extend([0, i])
        edges_type.extend([2, 2])

    # -- Pass 5: Static analysis flags (§16) ----------------------------------
    # ptr_to_id is fully populated for all instruction nodes. Detect absence
    # patterns and set node_flags[nid] = 1.0 for suspicious instructions.
    static_flags = _detect_static_flags(target_fn, ptr_to_id)
    for nid, flag in static_flags.items():
        if nid < len(node_flags):
            node_flags[nid] = flag

    # -------------------------------------------------------------------------
    x          = np.column_stack([
                     np.array(node_opcodes,    dtype=np.float32),
                     np.array(node_magnitudes, dtype=np.float32),
                     np.array(node_flags,      dtype=np.float32),
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
        dst = DATA / f"{split}_instr_v5_graphs.pkl"
        if not src.exists():
            print(f"Missing {src} -- run preprocess.py or drop --skip-download.")
            sys.exit(1)
        print(f"\n-- {split} ---------------------------------------------------")
        graphs = process_split_instr(src, subset=args.subset,
                                      workers=args.workers, seed=args.seed)
        with open(dst, "wb") as f:
            pickle.dump(graphs, f)
        print(f"  Saved {len(graphs)} graphs -> {dst}")

    print("\nDone. Run train_instr_v5.py next.\n")


if __name__ == "__main__":
    main()
