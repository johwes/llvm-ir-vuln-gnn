#!/usr/bin/env python3
"""
preprocess_instr.py — Instruction-level LLVM IR graphs from Devign.

Each instruction becomes a node (~300-500 nodes per function vs ~15 blocks).
A 3-line patch changes 3-5 nodes directly, rather than leaving the block-level
adjacency matrix functionally invariant — the root cause of the ~58% ceiling
documented in docs/experiments/ir-embed.md.

Outputs: data/{train,valid,test}_instr_graphs.pkl
  Each graph dict: {"x": int64 (N,1), "edge_index": int64 (2,E),
                    "edge_type": int64 (E,), "y": int, "idx": int}

Edge types: 0=CFG (sequential + inter-block), 1=DFG (SSA def-use),
            2=Global (bidirectional to virtual context node ID=0)

Usage:
    python preprocess_instr.py --subset 200   # smoke test (~100 graphs survive)
    python preprocess_instr.py                # full Devign dataset
    python preprocess_instr.py --workers 8    # parallel compilation
"""

import argparse
import ctypes
import json
import pickle
import random
import re
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
# Opcode vocabulary — 110 entries, indices 0-109
#
# 0     Virtual Context Node (not an opcode; added as node 0 in every graph)
# 1     Function Argument    (not an opcode; one node per argument)
# 2-23  ALU / vector
# 26-32 Memory
# 36-43 Control-flow
# 46    icmp fallback (unrecognised predicate)
# 47    fcmp fallback (unrecognised predicate)
# 48-60 Cast
# 61-68 Other
# 75    Mock/Global (external function call target or global variable)
# 76    Constant integer literal
# 77    Constant floating-point literal
# 78    Undef / poison value
# 79    Unknown fallback
# 80-89 icmp predicates: eq ne slt sle sgt sge ult ule ugt uge
# 90-105 fcmp predicates: false oeq ogt oge olt ole one ord uno ueq ugt uge ult ule une true
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
    # Comparison — fallback IDs used when predicate is unrecognised.
    # _instr_node_id() routes known predicates to IDs 80-105 instead.
    "icmp": 46, "fcmp": 47,
    # Cast
    "trunc": 48, "zext": 49, "sext": 50, "fptrunc": 51, "fpext": 52,
    "fptoui": 53, "fptosi": 54, "uitofp": 55, "sitofp": 56,
    "ptrtoint": 57, "inttoptr": 58, "bitcast": 59, "addrspacecast": 60,
    # Other
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

# icmp predicate → vocab ID (80-89)
_ICMP_PRED_IDS: dict[str, int] = {
    "eq": 80,  "ne": 81,
    "slt": 82, "sle": 83, "sgt": 84, "sge": 85,
    "ult": 86, "ule": 87, "ugt": 88, "uge": 89,
}
# fcmp predicate → vocab ID (90-105)
_FCMP_PRED_IDS: dict[str, int] = {
    "false": 90, "oeq": 91, "ogt": 92, "oge": 93,
    "olt":  94,  "ole": 95, "one": 96, "ord": 97,
    "uno":  98,  "ueq": 99, "ugt": 100, "uge": 101,
    "ult":  102, "ule": 103, "une": 104, "true": 105,
}


def _instr_node_id(instr) -> int:
    """Return vocab ID for an llvmlite instruction, expanding icmp/fcmp predicates."""
    op = instr.opcode
    if op == "icmp":
        m = _ICMP_PRED_RE.search(str(instr))
        if m:
            return _ICMP_PRED_IDS.get(m.group(1), IDX_UNKNOWN)
        return 46  # fallback: icmp with unrecognised predicate
    if op == "fcmp":
        m = _FCMP_PRED_RE.search(str(instr))
        if m:
            return _FCMP_PRED_IDS.get(m.group(1), IDX_UNKNOWN)
        return 47  # fallback: fcmp with unrecognised predicate
    return OPCODE_VOCAB.get(op, IDX_UNKNOWN)


# ValueKind integer constants (llvmlite 0.47 / LLVM 14)
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
    """Stable integer identity for an llvmlite value -- raw C++ pointer."""
    return ctypes.cast(v._ptr, ctypes.c_void_p).value


# ---------------------------------------------------------------------------
# Instruction-level graph extractor -- 5-pass algorithm
# ---------------------------------------------------------------------------

def ir_to_graph_instr(ir_text: str) -> dict | None:
    """
    Convert LLVM IR text to an instruction-level graph dict.

    Returns None if parsing fails or the result has fewer than 2 nodes.
    The caller adds "y" and "idx" to the returned dict.
    """
    try:
        mod = llvm.parse_assembly(ir_text)
    except Exception:
        return None

    # Pass 0 -- Find target function: last non-declaration in module.
    # The compiled IR contains inline helpers from the preamble headers before
    # the user's function; the user's function is always last.
    target_fn = None
    for fn in mod.functions:
        if not fn.is_declaration:
            target_fn = fn
    if target_fn is None:
        return None

    # -- Pass 1: Allocate nodes top-down ------------------------------------
    node_opcodes: list[int] = []
    ptr_to_id: dict[int, int] = {}
    node_counter = 0

    # Node 0: Virtual Context Node -- reduces graph diameter to O(1)
    node_opcodes.append(IDX_CONTEXT)
    node_counter = 1

    # One node per function argument (arguments are never "defined" by an
    # instruction; initialising them here prevents null-lookup crashes in Pass 3)
    for arg in target_fn.arguments:
        ptr_to_id[_ptr_id(arg)] = node_counter
        node_opcodes.append(IDX_ARGUMENT)
        node_counter += 1

    # One node per instruction; record each block's first instruction for
    # inter-block CFG edges (Pass 2 terminator handling)
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

    # -- Pass 2: CFG edges (type 0) -----------------------------------------
    for block in target_fn.blocks:
        prev_id = None
        instrs  = list(block.instructions)
        for instr in instrs:
            cur_id = ptr_to_id[_ptr_id(instr)]
            # Intra-block: sequential edge from previous instruction
            if prev_id is not None:
                edges_src.append(prev_id)
                edges_dst.append(cur_id)
                edges_type.append(0)
            prev_id = cur_id

        # Inter-block: terminator -> first instruction of successor block.
        # Terminator operands that are basic blocks carry the CFG successors.
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

    # -- Pass 3: DFG edges (type 1) -----------------------------------------
    # Caches deduplicate constant/global nodes.
    # LLVM interns constants -- same literal -> same _ptr -- so pointer-based
    # caching naturally prevents duplicate nodes for repeated constants (e.g.
    # the null sentinel 0 used 15 times -> 1 node, 15 DFG edges).
    constant_cache: dict[int, int] = {}  # ptr_id -> node_id
    mock_cache:     dict[str, int] = {}  # name   -> node_id

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
                        mock_cache[name] = node_counter
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

                # VK_BASIC_BLOCK: phi block operands -- skip (control, not data)
                # other ValueKinds: skip

    # -- Pass 4: Global Context edges (type 2) -- bidirectional -------------
    # Both directions are explicit rows so PyG's message passing works in
    # both directions without needing undirected-edge mode.
    # node_counter now includes constants and mocks added in Pass 3.
    for i in range(1, node_counter):
        edges_src.extend([i, 0])
        edges_dst.extend([0, i])
        edges_type.extend([2, 2])

    x          = np.array(node_opcodes, dtype=np.int64).reshape(-1, 1)
    edge_index = (np.array([edges_src, edges_dst], dtype=np.int64)
                  if edges_src else np.zeros((2, 0), dtype=np.int64))
    edge_type  = (np.array(edges_type, dtype=np.int64)
                  if edges_type else np.zeros(0, dtype=np.int64))

    return {"x": x, "edge_index": edge_index, "edge_type": edge_type}


# ---------------------------------------------------------------------------
# Per-item processing (called in parallel workers)
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
    ap.add_argument("--subset",       type=int, default=None,
                    help="N examples per split for a quick test")
    ap.add_argument("--workers",      type=int, default=4)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip download/split if data/*.jsonl already exist")
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
        dst = DATA / f"{split}_instr_graphs.pkl"
        if not src.exists():
            print(f"Missing {src} -- run preprocess.py or drop --skip-download.")
            sys.exit(1)
        print(f"\n-- {split} ---------------------------------------------------")
        graphs = process_split_instr(src, subset=args.subset,
                                      workers=args.workers, seed=args.seed)
        with open(dst, "wb") as f:
            pickle.dump(graphs, f)
        print(f"  Saved {len(graphs)} graphs -> {dst}")

    print("\nDone. Run train_instr.py next.\n")


if __name__ == "__main__":
    main()
