#!/usr/bin/env python3
"""
preprocess_juliet.py — §27 PDG backward-slice graphs from Juliet Test Suite.

§27 experiment: Pretrain on the Juliet Test Suite (NSA/NIST) to give the GNN
a clean structural signal before fine-tuning on Devign.

Why Juliet:
  - ~100 k synthetic C functions, paired bad/good per CWE
  - Structurally identical except at the bug site → zero label noise
  - Covers CWE-121 (stack overflow), CWE-122 (heap overflow), CWE-134 (format
    string), CWE-415 (double free), CWE-476 (null deref) — all detectable via
    structural sink-guard analysis
  - Trains the GNN to distinguish "guarded" from "unguarded" not "complex code
    from FFmpeg" vs "simple code from QEMU"

Node feature matrix: x shape (N, 3)  ← new vs §12's (N, 1)
  col 0: opcode_id      (int64, for nn.Embedding, 0-109)
  col 1: guard_class    (int64, 0=none, 1=bounds_check, 2=null_check)
  col 2: is_external_input (int64, 0 or 1)

  guard_class is set on icmp nodes in the slice:
    slt/sle/sgt/sge/ult/ule/ugt/uge → 1 (bounds check)
    eq/ne                           → 2 (null / equality check)
  is_external_input is set on call nodes whose callee is in INPUT_SOURCES.

These are the same features that slice_context.py exposes for LLM enrichment,
now baked into the graph for the GNN.

Output: data/{train,valid}_juliet_graphs.pkl
  (no separate test split — Devign valid/test serve as cross-dataset validation)

Juliet source:
  https://samate.nist.gov/SARD/test-suites/112
  File: Juliet_Test_Suite_v1.3_for_C_Cpp.zip  (~150 MB)

CWEs extracted (configurable via --cwes):
  CWE121  stack-based buffer overflow
  CWE122  heap-based buffer overflow
  CWE134  uncontrolled format string
  CWE415  double free
  CWE476  null pointer dereference

Usage:
    python preprocess_juliet.py --subset 4000 --workers 1   # smoke test
    python preprocess_juliet.py --workers 8                  # full extract
"""

import argparse
import ctypes
import os
import pickle
import random
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import llvmlite.binding as llvm

HERE = Path(__file__).parent
DATA = HERE / "data"

sys.path.insert(0, str(HERE))
from preprocess import compile_to_ir

JULIET_URL  = "https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip"
JULIET_ZIP  = DATA / "juliet_v1_3.zip"
JULIET_DIR  = DATA / "juliet_src"

TARGET_CWES = {"CWE121", "CWE122", "CWE134", "CWE415", "CWE476"}

# ---------------------------------------------------------------------------
# Opcode vocabulary (identical to preprocess_slice_pdg.py)
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

# guard_class values (column 1 in x)
GUARD_NONE   = 0
GUARD_BOUNDS = 1   # slt sle sgt sge ult ule ugt uge
GUARD_NULL   = 2   # eq ne

_BOUNDS_PREDS = frozenset({"slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"})
_NULL_PREDS   = frozenset({"eq", "ne"})

VK_ARGUMENT     = 0
VK_BASIC_BLOCK  = 1
VK_FUNCTION     = 5
VK_GLOBAL_VAR   = 8
VK_UNDEF        = 14
VK_CONSTANT_INT = 18
VK_CONSTANT_FP  = 19
VK_INSTRUCTION  = 24
VK_POISON       = 25

DANGEROUS_SINKS = frozenset({
    "strcpy", "strncpy", "strcat", "strncat",
    "memcpy", "memmove", "memset", "bcopy",
    "sprintf", "snprintf", "vsprintf", "vsnprintf",
    "gets", "fgets", "scanf", "sscanf", "fscanf",
    "read", "recv", "recvfrom", "pread",
    "malloc", "calloc", "realloc", "free", "xmalloc", "xrealloc",
    "printf", "fprintf", "syslog", "err", "warn",
    # Integer conversion — unchecked return used as array index / size is a
    # common vulnerability pattern (scar_atoi in scarnet, CWE-190/191).
    "atoi", "atol", "atoll", "atof",
    "strtol", "strtoul", "strtoll", "strtoull", "strtod",
})

INPUT_SOURCES = frozenset({
    "read", "recv", "recvfrom", "pread",
    "fgets", "fread", "getline", "getdelim",
    "scanf", "sscanf", "fscanf", "gets",
})

_SINK_SUFFIXES = tuple(DANGEROUS_SINKS)
_CONSTANT_IDS  = frozenset({IDX_CONST_INT, IDX_CONST_FP, IDX_UNDEF, IDX_CONTEXT})


def _ptr_id(v) -> int:
    return ctypes.cast(v._ptr, ctypes.c_void_p).value


def _icmp_pred(instr_text: str) -> str | None:
    m = _ICMP_PRED_RE.search(instr_text)
    return m.group(1) if m else None


def _instr_opcode_id(instr) -> int:
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


def _guard_class_for_opcode(opcode_id: int) -> int:
    if opcode_id in {82, 83, 84, 85, 86, 87, 88, 89}:  # slt sle sgt sge ult ule ugt uge
        return GUARD_BOUNDS
    if opcode_id in {80, 81}:  # eq ne
        return GUARD_NULL
    return GUARD_NONE


def _is_dangerous(name: str) -> bool:
    name = name.lstrip("@")
    if name in DANGEROUS_SINKS:
        return True
    for s in _SINK_SUFFIXES:
        if name.endswith(s) or name.endswith("_" + s):
            return True
    for s in ("memcpy", "memmove", "memset", "bcopy"):
        if name.startswith(f"llvm.{s}."):
            return True
    return False


def _canonical_name(name: str) -> str:
    name = name.lstrip("@")
    if name in DANGEROUS_SINKS:
        return name
    for s in ("memcpy", "memmove", "memset", "bcopy"):
        if name.startswith(f"llvm.{s}."):
            return s
    for s in _SINK_SUFFIXES:
        if name.endswith(s) or name.endswith("_" + s):
            return s
    return name


# ---------------------------------------------------------------------------
# PDG backward slice — multi-feature version (x shape: N × 3)
# ---------------------------------------------------------------------------

def _extract_slice_pdg_v7(x3, edge_index, edge_type, mock_names,
                           instr_to_block, block_preds, block_last_instr):
    """
    PDG backward slice identical to §12 but produces x shape (N, 3).

    x3[:, 0] = opcode_id (already in pre-built x3)
    x3[:, 1] = guard_class — 0/1/2, set for icmp nodes
    x3[:, 2] = is_external_input — 1 for call nodes whose callee is INPUT_SOURCES

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

    dangerous_mocks = {nid for nid, nm in mock_names.items() if _is_dangerous(nm)}
    source_mocks    = {nid for nid, nm in mock_names.items()
                       if _canonical_name(nm) in INPUT_SOURCES}

    sink_ids:    set[int]       = set()
    sink_to_fn:  dict[int, str] = {}
    for mid in dangerous_mocks:
        for consumer in fwd_dfg[mid]:
            if int(x3[consumer, 0]) == 63:  # call
                sink_ids.add(consumer)
                sink_to_fn[consumer] = _canonical_name(mock_names[mid])

    for i in range(E):
        if int(edge_type[i]) == 1:
            s, d = int(edge_index[0, i]), int(edge_index[1, i])
            if int(x3[d, 0]) in (29, 26) and int(x3[s, 0]) not in _CONSTANT_IDS:
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
    slice_size  = len(slice_nodes) + 1
    old_to_new  = {old: new + 1 for new, old in enumerate(slice_nodes)}

    if slice_size < 2:
        return None

    # Build new x3 with 3 columns
    new_x = np.zeros((slice_size, 3), dtype=np.int64)
    new_x[0, 0] = IDX_CONTEXT  # context node: guard_class=0, is_ext=0

    for new_id, old_id in enumerate(slice_nodes, start=1):
        opcode_id = int(x3[old_id, 0])
        new_x[new_id, 0] = opcode_id
        new_x[new_id, 1] = _guard_class_for_opcode(opcode_id)
        # is_external_input: call node whose callee mock is in INPUT_SOURCES
        if opcode_id == 63:  # call
            for pred in rev_dfg[old_id]:
                if pred in source_mocks and pred in old_to_new:
                    new_x[new_id, 2] = 1
                    break

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
            "sink_fn_names": {old_to_new[k]: v for k, v in sink_to_fn.items()
                              if k in old_to_new},
            "_sliced": True, "_n_sinks": len(sink_ids)}


def ir_to_graph_slice_pdg_v7(ir_text: str, fn_name: str | None = None):
    """
    Build instruction-level graph then extract PDG backward slice.

    Produces x shape (N, 3): [opcode_id, guard_class, is_external_input].
    fn_name: select a specific function from a multi-function module.
    Returns None on parse failure or slice < 2 nodes.
    """
    try:
        mod = llvm.parse_assembly(ir_text)
    except Exception:
        return None

    target_fn = None
    for fn in mod.functions:
        if fn.is_declaration:
            continue
        if fn_name is None:
            target_fn = fn
        elif fn.name == fn_name:
            target_fn = fn
            break
    if target_fn is None:
        return None

    node_opcodes:   list[int]     = []
    ptr_to_id:      dict[int,int] = {}
    instr_to_block: dict[int,int] = {}
    node_counter = 0

    node_opcodes.append(IDX_CONTEXT)
    node_counter = 1

    for arg in target_fn.arguments:
        ptr_to_id[_ptr_id(arg)] = node_counter
        node_opcodes.append(IDX_ARGUMENT)
        node_counter += 1

    block_first_instr: dict[int,int] = {}
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
            node_opcodes.append(_instr_opcode_id(instr))
            node_counter += 1

    if node_counter < 2:
        return None

    edges_src:  list[int] = []
    edges_dst:  list[int] = []
    edges_type: list[int] = []

    block_preds:      dict[int, list[int]] = defaultdict(list)
    block_last_instr: dict[int, int]       = {}

    for block in target_fn.blocks:
        bpid   = _ptr_id(block)
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
            for op in instrs[-1].operands:
                if op.value_kind == VK_BASIC_BLOCK:
                    succ_bpid  = _ptr_id(op)
                    succ_first = block_first_instr.get(succ_bpid)
                    if succ_first is not None:
                        edges_src.append(ptr_to_id[_ptr_id(instrs[-1])])
                        edges_dst.append(succ_first)
                        edges_type.append(0)
                    block_preds[succ_bpid].append(bpid)

    constant_cache: dict[int, int] = {}
    mock_cache:     dict[str, int] = {}
    mock_names:     dict[int, str] = {}

    for block in target_fn.blocks:
        for instr in block.instructions:
            dst_id = ptr_to_id[_ptr_id(instr)]
            for op in instr.operands:
                vk = op.value_kind
                if vk in (VK_INSTRUCTION, VK_ARGUMENT):
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

    for i in range(1, node_counter):
        edges_src.extend([i, 0])
        edges_dst.extend([0, i])
        edges_type.extend([2, 2])

    # Build x3 (N, 3) — guard_class and is_external_input computed in slice fn
    x3 = np.array(node_opcodes, dtype=np.int64).reshape(-1, 1)
    x3 = np.hstack([x3, np.zeros((len(node_opcodes), 2), dtype=np.int64)])

    edge_index = (np.array([edges_src, edges_dst], dtype=np.int64)
                  if edges_src else np.zeros((2, 0), dtype=np.int64))
    edge_type  = (np.array(edges_type, dtype=np.int64)
                  if edges_type else np.zeros(0, dtype=np.int64))

    g = _extract_slice_pdg_v7(x3, edge_index, edge_type, mock_names,
                               instr_to_block, block_preds, block_last_instr)
    if g is None:
        # Fallback: full graph with 3-column x, guard_class computed per node
        new_x = np.zeros((node_counter, 3), dtype=np.int64)
        for i in range(node_counter):
            oc = int(x3[i, 0])
            new_x[i, 0] = oc
            new_x[i, 1] = _guard_class_for_opcode(oc)
        g = {"x": new_x, "edge_index": edge_index, "edge_type": edge_type,
             "sink_fn_names": {}, "_sliced": False, "_n_sinks": 0}

    return g


# ---------------------------------------------------------------------------
# Juliet C source extraction
# ---------------------------------------------------------------------------

# Match function *definitions* (not call sites) — Juliet names follow
# the pattern  CWE<N>_<Name>__<variant>_<NNN>_{bad,good,goodG2B,...}
# Capture group is the function name; we match the opening brace of the body.
_FN_DEF_RE = re.compile(
    r'void\s+(CWE\d+_\w+__\w+_\d+_(bad|good\w*))\s*\([^)]*\)\s*\{',
    re.MULTILINE,
)

_C_SUFFIX = ".c"

# ---------------------------------------------------------------------------
# Juliet preamble — inline equivalent of std_testcase.h + common stubs
#
# Juliet sources all do:
#   #include "std_testcase.h"          (local header, not on system path)
#   #include "std_testcase_helper.h"   (optional)
#   #include "CWE121_*.h"              (CWE-specific helpers)
#
# Strategy (same as Devign's compile_to_ir): strip local #include "..." lines,
# prepend our inline preamble, and rely on compile_to_ir()'s iterative stub
# injector to fix any remaining unknowns from clang stderr.
# ---------------------------------------------------------------------------

# Juliet-specific stubs only — NO macro redefinitions.
# compile_to_ir() from preprocess.py already prepends PREAMBLE which:
#   - includes <stdio.h>, <stdlib.h>, <string.h>, <stdint.h>, <stddef.h>
#   - defines #define static, #define inline, #define __attribute__(x), etc.
#   - calls _strip_asm_blocks() on our text — so we MUST NOT put any
#     __asm__ token here or it will be corrupted to ((void)0).
_JULIET_STUBS = """\
#include <wchar.h>

/* std_testcase.h inline — Juliet function stubs */
void printLine(const char *l)        { (void)l; }
void printIntLine(int l)             { (void)l; }
void printLongLongLine(long long l)  { (void)l; }
void printLongLine(long l)           { (void)l; }
void printSizeTLine(size_t l)        { (void)l; }
void printUnsignedLine(unsigned l)   { (void)l; }
void printHexCharLine(char c)        { (void)c; }
void printWLine(const wchar_t *l)    { (void)l; }
void printWcharLine(wchar_t c)       { (void)c; }
void printFloatLine(float l)         { (void)l; }
void printDoubleLine(double l)       { (void)l; }

/* Flow-control helpers used by Juliet numbered variants */
int globalReturnsTrue(void)   { return 1; }
int globalReturnsFalse(void)  { return 0; }
int staticReturnsTrue(void)   { return 1; }
int staticReturnsFalse(void)  { return 0; }
int globalTrue  = 1;
int globalFalse = 0;
"""

_LOCAL_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"[^"]*"\s*$', re.MULTILINE)


# ---------------------------------------------------------------------------
# Pair extraction from zip — one entry per function definition
# ---------------------------------------------------------------------------

def _extract_juliet_pairs(zip_path: Path,
                           target_cwes: set[str],
                           max_per_cwe: int | None = None) -> list[dict]:
    """
    Walk the Juliet zip, collect one dict per bad/good function *definition*.

    Returns list of dicts:
      {"src_text": str, "fn_name": str, "target": int (0/1)}

    src_text is the raw C source with local #include "..." lines already
    stripped — compile_to_ir() handles any remaining unknowns via its
    iterative stub injector.
    """
    pairs: list[dict] = []
    cwe_counts: dict[str, int] = defaultdict(int)

    print(f"  Scanning {zip_path.name} for CWEs: {sorted(target_cwes)} ...")

    with zipfile.ZipFile(zip_path) as zf:
        c_names = [n for n in zf.namelist() if n.endswith(_C_SUFFIX)]

        for zname in c_names:
            parts = Path(zname).parts

            # Skip support/ files (headers / helper stubs)
            if "support" in parts:
                continue

            # Identify CWE
            cwe = None
            for part in parts:
                if part.startswith("CWE"):
                    candidate = part.split("_")[0]
                    if candidate in target_cwes:
                        cwe = candidate
                        break
            if cwe is None:
                continue

            if max_per_cwe and cwe_counts[cwe] >= max_per_cwe:
                continue

            try:
                src_text = zf.read(zname).decode(errors="replace")
            except Exception:
                continue

            # Strip local includes before storing — saves re-reading the zip
            stripped = _LOCAL_INCLUDE_RE.sub("", src_text)

            for m in _FN_DEF_RE.finditer(src_text):
                fn_name = m.group(1)
                is_bad  = "_bad" in fn_name and not fn_name.endswith("_bad_sink")
                label   = 1 if is_bad else 0
                pairs.append({"src_text": stripped, "fn_name": fn_name, "target": label})
                cwe_counts[cwe] += 1

    print(f"  Found {len(pairs)} function definitions across CWEs:")
    for cwe in sorted(cwe_counts):
        print(f"    {cwe}: {cwe_counts[cwe]}")
    return pairs


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------

def process_juliet_item(item: dict) -> dict | None:
    """
    Compile one Juliet C source (local includes already stripped) to IR,
    then build the PDG slice graph for the named function.

    Uses compile_to_ir() from preprocess.py — the same iterative stub
    injector used for Devign — with a Juliet-specific preamble that
    provides std_testcase.h symbols inline.
    """
    src_text = item["src_text"]
    fn_name  = item["fn_name"]

    full_source = _JULIET_STUBS + "\n" + src_text
    ir = compile_to_ir(full_source)
    if ir is None:
        return None

    g = ir_to_graph_slice_pdg_v7(ir, fn_name=fn_name)
    if g is None:
        return None

    g["y"]       = int(item["target"])
    g["fn_name"] = fn_name
    return g


# ---------------------------------------------------------------------------
# Download Juliet
# ---------------------------------------------------------------------------

# NIST SARD blocks plain urllib (no User-Agent). Try curl → wget → urllib
# with a browser UA in that order.  If all fail, print manual instructions.

def download_juliet(zip_path: Path) -> None:
    if zip_path.exists():
        print(f"  {zip_path.name} already present, skipping download.")
        return

    print(f"  Downloading Juliet Test Suite (~150 MB) ...")
    print(f"  URL: {JULIET_URL}")
    print("  (This may take a few minutes on a slow connection.)")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_suffix(".tmp")

    # 1. curl
    try:
        ret = subprocess.run(
            ["curl", "-L", "--fail", "-A",
             "Mozilla/5.0 (compatible; juliet-downloader/1.0)",
             "-o", str(tmp), JULIET_URL],
            check=True,
        )
        tmp.rename(zip_path)
        print(f"\n  Downloaded via curl → {zip_path}")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 2. wget
    try:
        subprocess.run(
            ["wget", "-q", "--show-progress",
             "-U", "Mozilla/5.0 (compatible; juliet-downloader/1.0)",
             "-O", str(tmp), JULIET_URL],
            check=True,
        )
        tmp.rename(zip_path)
        print(f"\n  Downloaded via wget → {zip_path}")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 3. urllib with browser User-Agent
    import urllib.request as _ureq
    req = _ureq.Request(
        JULIET_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; juliet-downloader/1.0)"},
    )
    try:
        with _ureq.urlopen(req) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length", 0))
            done  = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"    {done/total*100:.0f}%", end="\r", flush=True)
        tmp.rename(zip_path)
        print(f"\n  Downloaded via urllib → {zip_path}")
        return
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        print(f"\n  urllib failed: {exc}")

    # All methods failed — print manual instructions and exit
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║  Automatic download failed (NIST SARD requires browser).    ║")
    print("  ║                                                              ║")
    print("  ║  Manual download:                                            ║")
    print(f"  ║    {JULIET_URL[:58]}  ║")
    print("  ║                                                              ║")
    print(f"  ║  Save to: {str(zip_path)[:54]}  ║")
    print("  ║                                                              ║")
    print("  ║  Then re-run with --skip-download                           ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--subset",       type=int,  default=None,
                    help="Limit total items (balanced bad/good) for smoke test")
    ap.add_argument("--max-per-cwe",  type=int,  default=None,
                    help="Max items per CWE (applied before --subset)")
    ap.add_argument("--workers",      type=int,  default=4)
    ap.add_argument("--seed",         type=int,  default=42)
    ap.add_argument("--valid-frac",   type=float, default=0.1,
                    help="Fraction of data to hold out as validation set")
    ap.add_argument("--cwes",         type=str,
                    default="CWE121,CWE122,CWE134,CWE415,CWE476",
                    help="Comma-separated CWE IDs to extract")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--debug-first", action="store_true",
                    help="Show compile_to_ir stderr and graph result for the first pair, then exit")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    target_cwes = set(args.cwes.split(","))

    if not args.skip_download:
        download_juliet(JULIET_ZIP)

    if not JULIET_ZIP.exists():
        print(f"ERROR: {JULIET_ZIP} not found.")
        print(f"       Download from: {JULIET_URL}")
        sys.exit(1)

    print("\n-- Extract Juliet function pairs ----------------------------------------")
    pairs = _extract_juliet_pairs(JULIET_ZIP, target_cwes, args.max_per_cwe)

    if args.debug_first:
        import preprocess as _pre
        item = pairs[0]
        print(f"\n-- debug-first: fn_name={item['fn_name']}  label={item['target']}")
        print(f"-- Source (first 60 lines after strip):")
        for i, ln in enumerate(item["src_text"].splitlines()[:60], 1):
            print(f"  {i:3d}: {ln}")

        full = _JULIET_STUBS + "\n" + item["src_text"]
        cap: list[str] = []
        ir = _pre.compile_to_ir(full, _failure_capture=cap)
        if ir is None:
            print(f"\n-- compile_to_ir FAILED. Final stderr:")
            print(cap[0][:4000] if cap else "(no stderr captured)")
        else:
            print(f"\n-- compile_to_ir OK ({len(ir)} chars)")
            g = ir_to_graph_slice_pdg_v7(ir, fn_name=item["fn_name"])
            if g is None:
                print(f"-- ir_to_graph_slice_pdg_v7 returned None")
                print(f"   (function '{item['fn_name']}' not found or no sinks)")
                # Check if function exists in IR
                import llvmlite.binding as _llvm
                try:
                    mod = _llvm.parse_assembly(ir)
                    fns = [f.name for f in mod.functions if not f.is_declaration]
                    print(f"   Functions in IR: {fns[:20]}")
                except Exception as e:
                    print(f"   IR parse error: {e}")
            else:
                print(f"-- graph OK: x={g['x'].shape}  edges={g['edge_index'].shape}  "
                      f"sliced={g.get('_sliced')}  n_sinks={g.get('_n_sinks')}")
        sys.exit(0)

    rng = random.Random(args.seed)
    if args.subset:
        bad  = [p for p in pairs if p["target"] == 1]
        good = [p for p in pairs if p["target"] == 0]
        rng.shuffle(bad); rng.shuffle(good)
        half = args.subset // 2
        pairs = bad[:half] + good[:half]

    rng.shuffle(pairs)

    n_valid    = max(1, int(len(pairs) * args.valid_frac))
    valid_raw  = pairs[:n_valid]
    train_raw  = pairs[n_valid:]

    print(f"\n  Total pairs: {len(pairs)}  "
          f"(train={len(train_raw)}, valid={len(valid_raw)})")

    def _process_split(items: list[dict], split: str) -> list[dict]:
        graphs, ok, fail = [], 0, 0
        total = len(items)
        print(f"\n-- {split} ({total} items, {args.workers} workers) --")

        if args.workers == 1:
            for i, item in enumerate(items, 1):
                g = process_juliet_item(item)
                if g:
                    graphs.append(g); ok += 1
                else:
                    fail += 1
                if i % 200 == 0:
                    print(f"  {i}/{total}  ok={ok}  fail={fail}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(process_juliet_item, it): it for it in items}
                for i, fut in enumerate(as_completed(futs), 1):
                    g = fut.result()
                    if g:
                        graphs.append(g); ok += 1
                    else:
                        fail += 1
                    if i % 200 == 0:
                        print(f"  {i}/{total}  ok={ok}  fail={fail}")

        attrition = fail / total * 100 if total > 0 else 0
        print(f"  Done: {ok} graphs built, {fail} failed ({attrition:.0f}% attrition)")

        if graphs:
            nc  = [g["x"].shape[0] for g in graphs]
            ns  = sum(1 for g in graphs if g.get("_sliced", False))
            nv  = sum(1 for g in graphs if g["y"] == 1)
            print(f"  Slice stats: mean={np.mean(nc):.0f}  "
                  f"median={int(np.median(nc))}  max={max(nc)}")
            print(f"  Sliced: {ns}/{ok} ({100*ns/ok:.0f}%)")
            print(f"  Label balance: {nv} vuln / {ok-nv} benign")

            print(f"  Feature check — x shape: {graphs[0]['x'].shape}  "
                  f"(expect (N, 3))")
            guard_1 = sum(1 for g in graphs if np.any(g["x"][:, 1] == 1))
            guard_2 = sum(1 for g in graphs if np.any(g["x"][:, 1] == 2))
            ext_inp = sum(1 for g in graphs if np.any(g["x"][:, 2] == 1))
            print(f"  Graphs with bounds-check node:  {guard_1}")
            print(f"  Graphs with null-check node:    {guard_2}")
            print(f"  Graphs with external-input node: {ext_inp}")

        for g in graphs:
            g.pop("_sliced", None)
            g.pop("_n_sinks", None)

        return graphs

    train_graphs = _process_split(train_raw, "train")
    valid_graphs = _process_split(valid_raw, "valid")

    train_out = DATA / "train_juliet_graphs.pkl"
    valid_out = DATA / "valid_juliet_graphs.pkl"

    with open(train_out, "wb") as f:
        pickle.dump(train_graphs, f)
    with open(valid_out, "wb") as f:
        pickle.dump(valid_graphs, f)

    print(f"\nSaved:")
    print(f"  {train_out}  ({len(train_graphs)} graphs)")
    print(f"  {valid_out}  ({len(valid_graphs)} graphs)")
    print(f"\nNext: python train_slice_pdg_v7.py")


if __name__ == "__main__":
    main()
