#!/usr/bin/env python3
"""
preprocess_slice_pdg_v2.py — §22 PDG backward-slice graphs with taint flags.

Extends §12 (preprocess_slice_pdg.py) by adding intra-procedural taint
propagation as a second node feature column.

Motivation: §12 PDG slices already select the right subgraph (dangerous sink
+ its data/control dependencies). The missing discriminative signal is WHICH
nodes within that slice are directly suspicious vs. passively included.
Taint flags mark sources of dangerous values and propagate that signal forward
through DFG edges, giving the GNN an explicit annotation to reason about.

Taint sources (same patterns as §17 preprocess_instr_v6.py):

  Pattern A — dangerous call without icmp guard:
    Call to a dangerous function (DANGEROUS_SINKS) where neither the current
    block nor any CFG predecessor block contains an icmp instruction.

  Pattern B — unchecked return value:
    Call to an alloc/IO/network function (PATTERN_B_NAMES) whose return value
    is never used as an operand of any icmp in the function.

Taint propagation:
  BFS forward through DFG edges from each source node.
  Value decays by 0.5 per hop; stops at value < 0.05 or 3 hops.
  Each node takes the maximum taint from any reaching path.

Node feature matrix: (N, 2) float32
  x[:, 0] = opcode_id      (cast to long for nn.Embedding)
  x[:, 1] = taint_value    (0.0 = clean, 1.0 = direct source, 0.5/0.25 = propagated)

Output: data/{train,valid,test}_slice_pdg_v2_graphs.pkl

Usage:
    python preprocess_slice_pdg_v2.py --subset 200 --workers 1   # smoke test
    python preprocess_slice_pdg_v2.py --skip-download             # full Devign
    python preprocess_slice_pdg_v2.py --workers 8 --skip-download
"""

import argparse
import ctypes
import json
import pickle
import random
import re
import sys
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import llvmlite.binding as llvm

HERE = Path(__file__).parent
DATA = HERE / "data"

sys.path.insert(0, str(HERE))
from preprocess import compile_to_ir, download_devign

# ---------------------------------------------------------------------------
# Opcode vocabulary (identical to preprocess_slice_pdg.py — 110 entries)
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
# Dangerous sink patterns (identical to preprocess_slice_pdg.py)
# ---------------------------------------------------------------------------

DANGEROUS_SINKS = frozenset({
    "strcpy", "strncpy", "strcat", "strncat",
    "memcpy", "memmove", "memset", "bcopy",
    "sprintf", "snprintf", "vsprintf", "vsnprintf",
    "gets", "fgets", "scanf", "sscanf", "fscanf",
    "read", "recv", "recvfrom", "pread",
    "malloc", "calloc", "realloc", "free", "xmalloc", "xrealloc",
    "printf", "fprintf", "syslog", "err", "warn",
})

_SINK_SUFFIXES = tuple(DANGEROUS_SINKS)


def _is_dangerous(name: str) -> bool:
    name = name.lstrip("@")
    if name in DANGEROUS_SINKS:
        return True
    for s in _SINK_SUFFIXES:
        if name.endswith(s) or name.endswith("_" + s):
            return True
    return False


# ---------------------------------------------------------------------------
# §22: Taint source detection + propagation
# ---------------------------------------------------------------------------

# Pattern B: functions whose unchecked return value is dangerous
# (alloc returns NULL on failure; IO/network returns -1 on error)
_PATTERN_B_NAMES = frozenset({
    "malloc", "calloc", "realloc", "xmalloc", "xrealloc",
    "kmalloc", "kzalloc", "vmalloc",
    "fopen", "open", "read", "pread", "fgets", "gets", "fread",
    "recv", "recvfrom", "accept",
})


def _detect_taint_sources(target_fn, ptr_to_id: dict) -> dict[int, float]:
    """
    Detect Pattern A and B taint source nodes.
    Returns {node_id: 1.0} for each flagged call instruction.

    Pattern A — dangerous call without icmp guard (same as §17):
      A call to DANGEROUS_SINKS where neither the current block nor any
      CFG predecessor block contains an icmp instruction.

    Pattern B — unchecked return value (same as §17):
      A call to _PATTERN_B_NAMES whose return value is never used as an
      operand of any icmp instruction anywhere in the function.
    """
    flags: dict[int, float] = {}

    # Pattern B: collect candidates then find icmp uses
    candidate_ptrs: dict[int, int] = {}  # ptr_id → node_id
    for block in target_fn.blocks:
        for instr in block.instructions:
            if instr.opcode != "call":
                continue
            for op in instr.operands:
                if op.value_kind in (VK_GLOBAL_VAR, VK_FUNCTION):
                    if op.name.lstrip("@") in _PATTERN_B_NAMES:
                        nid = ptr_to_id.get(_ptr_id(instr))
                        if nid is not None:
                            candidate_ptrs[_ptr_id(instr)] = nid

    checked_ptrs: set[int] = set()
    if candidate_ptrs:
        for block in target_fn.blocks:
            for instr in block.instructions:
                if instr.opcode == "icmp":
                    for op in instr.operands:
                        if _ptr_id(op) in candidate_ptrs:
                            checked_ptrs.add(_ptr_id(op))

    for ptr, nid in candidate_ptrs.items():
        if ptr not in checked_ptrs:
            flags[nid] = 1.0

    # Pattern A: build CFG predecessor map, then flag unguarded dangerous calls
    block_has_icmp: dict[int, bool] = {}
    block_preds:    dict[int, list[int]] = {}

    for block in target_fn.blocks:
        bpid = _ptr_id(block)
        block_has_icmp[bpid] = any(i.opcode == "icmp" for i in block.instructions)
        block_preds[bpid] = []

    for block in target_fn.blocks:
        instrs = list(block.instructions)
        if instrs:
            for op in instrs[-1].operands:
                if op.value_kind == VK_BASIC_BLOCK:
                    succ_pid = _ptr_id(op)
                    if succ_pid in block_preds:
                        block_preds[succ_pid].append(_ptr_id(block))

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
            for op in instr.operands:
                if op.value_kind in (VK_GLOBAL_VAR, VK_FUNCTION):
                    if _is_dangerous(op.name):
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
    """BFS forward through DFG edges from flagged sources with exponential decay."""
    fwd_dfg: dict[int, list[int]] = {}
    for s, d, t in zip(edges_src, edges_dst, edges_type):
        if t == 1:
            fwd_dfg.setdefault(s, []).append(d)

    result = dict(flags)
    for start, start_val in flags.items():
        queue: deque[tuple[int, float, int]] = deque([(start, start_val, 0)])
        seen: set[int] = {start}
        while queue:
            node, val, hop = queue.popleft()
            if hop >= max_hops:
                continue
            next_val = val * decay
            if next_val < min_val:
                continue
            for nb in fwd_dfg.get(node, []):
                if nb not in seen:
                    seen.add(nb)
                    result[nb] = max(result.get(nb, 0.0), next_val)
                    queue.append((nb, next_val, hop + 1))
    return result


# ---------------------------------------------------------------------------
# PDG backward slice extractor — v2 (with taint)
# ---------------------------------------------------------------------------

_CONSTANT_IDS = frozenset({IDX_CONST_INT, IDX_CONST_FP, IDX_UNDEF, IDX_CONTEXT})


def _extract_slice_pdg_v2(x, edge_index, edge_type, mock_names,
                           instr_to_block, block_preds, block_last_instr,
                           taint_array: np.ndarray):
    """
    PDG backward slice (identical algorithm to §12) but outputs (N, 2) float32 x:
      new_x[:, 0] = opcode_id
      new_x[:, 1] = taint_value (from taint_array, keyed by old node id)

    Returns None if no dangerous sinks found.
    """
    E = edge_index.shape[1] if edge_index.ndim == 2 and edge_index.shape[1] > 0 else 0

    fwd_dfg: dict[int, list[int]] = defaultdict(list)
    rev_dfg: dict[int, list[int]] = defaultdict(list)
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
            if int(x[consumer, 0]) == 63:
                sink_ids.add(consumer)

    # Sink type 2: GEP with non-constant index
    for i in range(E):
        if int(edge_type[i]) == 1:
            s, d = int(edge_index[0, i]), int(edge_index[1, i])
            if int(x[d, 0]) == 29 and int(x[s, 0]) not in _CONSTANT_IDS:
                sink_ids.add(d)

    if not sink_ids:
        return None

    visited      = set(sink_ids)
    ctrl_checked: set[int] = set()

    changed = True
    while changed:
        changed = False

        frontier = list(visited)
        while frontier:
            nxt = []
            for node in frontier:
                for pred in rev_dfg[node]:
                    if pred not in visited and pred != 0:
                        visited.add(pred)
                        nxt.append(pred)
                        changed = True
            frontier = nxt

        new_nodes = visited - ctrl_checked
        ctrl_checked |= new_nodes
        for node in new_nodes:
            block_id = instr_to_block.get(node)
            if block_id is None:
                continue
            for pred_block in block_preds.get(block_id, []):
                term_id = block_last_instr.get(pred_block)
                if term_id is not None and term_id not in visited and term_id != 0:
                    visited.add(term_id)
                    changed = True

    slice_nodes = sorted(visited)
    slice_size  = len(slice_nodes) + 1  # +1 for context node
    old_to_new  = {old: new + 1 for new, old in enumerate(slice_nodes)}

    if slice_size < 2:
        return None

    # Build (N, 2) float32 feature matrix
    new_x = np.zeros((slice_size, 2), dtype=np.float32)
    new_x[0, 0] = float(IDX_CONTEXT)
    new_x[0, 1] = 0.0
    for new_id, old_id in enumerate(slice_nodes, start=1):
        new_x[new_id, 0] = float(int(x[old_id, 0]))
        new_x[new_id, 1] = float(taint_array[old_id]) if old_id < len(taint_array) else 0.0

    new_src, new_dst, new_et = [], [], []
    for i in range(E):
        et = int(edge_type[i])
        if et == 2:
            continue
        s, d = int(edge_index[0, i]), int(edge_index[1, i])
        if s in old_to_new and d in old_to_new:
            new_src.append(old_to_new[s])
            new_dst.append(old_to_new[d])
            new_et.append(et)

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
# Graph builder — 5-pass algorithm + PDG slice + taint
# ---------------------------------------------------------------------------

def ir_to_graph_slice_pdg_v2(ir_text: str) -> dict | None:
    """
    Build instruction-level PDG backward-slice graph with taint features.

    Passes 1–4: identical to preprocess_slice_pdg.py (§12).
    Pass 5: detect taint sources (Pattern A + B) and propagate through DFG.
    Slice extraction: same PDG algorithm but x is (N, 2) float32.

    Caller adds 'y' and 'idx'.
    """
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

    # -- Pass 1: allocate nodes + track instruction→block membership ----------
    node_opcodes   = []
    ptr_to_id      = {}
    instr_to_block = {}
    node_counter   = 0

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
            ptr_to_id[ipid]              = node_counter
            instr_to_block[node_counter] = bpid
            node_opcodes.append(_instr_node_id(instr))
            node_counter += 1

    if node_counter < 2:
        return None

    edges_src:  list[int] = []
    edges_dst:  list[int] = []
    edges_type: list[int] = []

    # -- Pass 2: CFG edges + predecessor/terminator maps ----------------------
    block_preds      = defaultdict(list)
    block_last_instr = {}

    for block in target_fn.blocks:
        bpid    = _ptr_id(block)
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
            block_last_instr[bpid] = ptr_to_id[_ptr_id(instrs[-1])]
            terminator = instrs[-1]
            term_id    = ptr_to_id[_ptr_id(terminator)]
            for op in terminator.operands:
                if op.value_kind == VK_BASIC_BLOCK:
                    succ_bpid  = _ptr_id(op)
                    succ_first = block_first_instr.get(succ_bpid)
                    if succ_first is not None:
                        edges_src.append(term_id)
                        edges_dst.append(succ_first)
                        edges_type.append(0)
                    block_preds[succ_bpid].append(bpid)

    # -- Pass 3: DFG edges + mock name tracking --------------------------------
    constant_cache: dict[int, int] = {}
    mock_cache:     dict[str, int] = {}
    mock_names:     dict[int, str] = {}

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
                        mock_cache[name]      = node_counter
                        mock_names[node_counter] = name
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

    # -- Pass 4: global context edges (type 2) — bidirectional ----------------
    for i in range(1, node_counter):
        edges_src.extend([i, 0])
        edges_dst.extend([0, i])
        edges_type.extend([2, 2])

    x          = np.array(node_opcodes, dtype=np.int64).reshape(-1, 1)
    edge_index = (np.array([edges_src, edges_dst], dtype=np.int64)
                  if edges_src else np.zeros((2, 0), dtype=np.int64))
    edge_type  = (np.array(edges_type, dtype=np.int64)
                  if edges_type else np.zeros(0, dtype=np.int64))

    # -- Pass 5: taint detection + propagation --------------------------------
    source_flags = _detect_taint_sources(target_fn, ptr_to_id)
    taint_map    = _propagate_taint(source_flags, edges_src, edges_dst, edges_type)
    taint_array  = np.zeros(node_counter, dtype=np.float32)
    for nid, val in taint_map.items():
        if nid < node_counter:
            taint_array[nid] = val

    # -- PDG slice extraction -------------------------------------------------
    g = _extract_slice_pdg_v2(x, edge_index, edge_type, mock_names,
                               instr_to_block, block_preds, block_last_instr,
                               taint_array)
    if g is None:
        # Fallback: full graph with taint as second column
        x_float = np.column_stack([
            x.astype(np.float32),
            taint_array[:node_counter].reshape(-1, 1),
        ])
        g = {"x": x_float, "edge_index": edge_index, "edge_type": edge_type,
             "_sliced": False, "_n_sinks": 0}

    return g


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------

def process_item(item: dict) -> dict | None:
    ir = compile_to_ir(item["func"])
    if ir is None:
        return None
    g = ir_to_graph_slice_pdg_v2(ir)
    if g is None:
        return None
    g["y"]   = int(item["target"])
    g["idx"] = item.get("idx", 0)
    return g


def process_split(jsonl_path: Path, subset: int | None,
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
            g = process_item(item)
            if g:
                graphs.append(g); ok += 1
            else:
                fail += 1
            if i % 500 == 0:
                print(f"    {i}/{total}  ok={ok}  failed={fail}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(process_item, it): it for it in items}
            for i, fut in enumerate(as_completed(futs), 1):
                g = fut.result()
                if g:
                    graphs.append(g); ok += 1
                else:
                    fail += 1
                if i % 500 == 0:
                    print(f"    {i}/{total}  ok={ok}  failed={fail}")

    attrition = fail / total * 100 if total > 0 else 0
    print(f"  Done: {ok} graphs built, {fail} failed ({attrition:.0f}% attrition)")

    node_counts = [g["x"].shape[0] for g in graphs]
    n_sliced    = sum(1 for g in graphs if g.get("_sliced", False))
    n_fallback  = ok - n_sliced
    n_tainted   = sum(1 for g in graphs if g["x"][:, 1].max() > 0)
    if node_counts:
        print(f"  Slice stats: mean={np.mean(node_counts):.0f} nodes  "
              f"median={int(np.median(node_counts))}  max={max(node_counts)}")
        print(f"  Sliced: {n_sliced}/{ok} ({100*n_sliced/ok:.0f}%)  "
              f"Fallback: {n_fallback}/{ok} ({100*n_fallback/ok:.0f}%)")
        print(f"  Taint coverage: {n_tainted}/{ok} graphs have ≥1 tainted node "
              f"({100*n_tainted/ok:.0f}%)")

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
        dst = DATA / f"{split}_slice_pdg_v2_graphs.pkl"
        if not src.exists():
            print(f"Missing {src} -- run preprocess.py or drop --skip-download.")
            sys.exit(1)
        print(f"\n-- {split} ---------------------------------------------------")
        graphs = process_split(src, subset=args.subset,
                               workers=args.workers, seed=args.seed)
        with open(dst, "wb") as f:
            pickle.dump(graphs, f)
        print(f"  Saved {len(graphs)} graphs -> {dst}")

    print("\nDone. Run train_slice_pdg_v2.py next.\n")


if __name__ == "__main__":
    main()
