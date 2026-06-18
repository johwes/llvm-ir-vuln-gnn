#!/usr/bin/env python3
"""
preprocess_joern.py — Extract PDG slice graphs from C/C++ source using Joern.

Unlike preprocess_slice_pdg.py (clang + llvmlite), this extractor uses Joern's
fuzzy "island grammar" parser — no compilation, no headers required.

Coverage: ~95% of PrimeVul functions vs ~34% for the clang pipeline.
Vulnerable functions no longer fail at 2x the rate of benign ones.

Node vocabulary: 16 Joern CPG node type tokens (vs 110 LLVM opcodes).
Edge types:      same 3-type scheme — CFG=0, REACHING_DEF=1, AST=2.
Output dict:     same format as preprocess_slice_pdg.py — drop-in for training.

Prerequisites:
    joern-parse and joern-export on PATH, or set JOERN_HOME env var.
    Default: ~/bin/joern/joern-cli/

Usage:
    python preprocess_joern.py path/to/function.c
    python preprocess_joern.py path/to/function.c --json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Joern binary locations
# ---------------------------------------------------------------------------

_JOERN_HOME  = Path(os.environ.get("JOERN_HOME",
                    Path.home() / "bin/joern/joern-cli"))
JOERN_PARSE  = _JOERN_HOME / "joern-parse"
JOERN_EXPORT = _JOERN_HOME / "joern-export"


# ---------------------------------------------------------------------------
# Node vocabulary
# ---------------------------------------------------------------------------
#
# CALL nodes are split into four sub-categories for discriminative power:
#   CALL_SINK       — dangerous function (memcpy, strcpy, malloc, ...)
#   CALL_COMPARISON — guard operator (<, >, ==, !=, &&, ||)  ← "icmp" equivalent
#   CALL_ALLOC      — allocation operator (<operator>.alloc)
#   CALL_OPERATOR   — any other <operator>.xxx
#   CALL_REGULAR    — regular named function call
#
# CALL_COMPARISON is the structural equivalent of the LLVM icmp opcode used in
# preprocess_slice_pdg.py to detect guards.  A slice with no CALL_COMPARISON
# node between the argument source and the sink is UNGUARDED.

NODE_VOCAB = {
    "METHOD_PARAMETER_IN":  0,   # function argument — primary taint source
    "METHOD_PARAMETER_OUT": 1,   # output parameter
    "IDENTIFIER":           2,   # variable use / reference
    "LOCAL":                3,   # local variable declaration
    "LITERAL":              4,   # numeric / string / char constant
    "FIELD_IDENTIFIER":     5,   # struct field access (ctx->cert)
    "CALL_SINK":            6,   # dangerous sink
    "CALL_COMPARISON":      7,   # guard: <, >, ==, !=, &&, ||
    "CALL_ALLOC":           8,   # <operator>.alloc / arrayInitializer
    "CALL_OPERATOR":        9,   # other <operator>.xxx
    "CALL_REGULAR":         10,  # regular named call
    "CONTROL_STRUCTURE":    11,  # if / while / for / do / switch
    "JUMP_TARGET":          12,  # label / jump target
    "METHOD_RETURN":        13,  # return node
    "BLOCK":                14,  # basic block / compound statement
    "UNKNOWN":              15,  # fallback
}

VOCAB_SIZE = len(NODE_VOCAB)   # 16

# Edge type indices
EDGE_CFG          = 0
EDGE_REACHING_DEF = 1
EDGE_AST          = 2

# ---------------------------------------------------------------------------
# Dangerous sinks (same set as preprocess_slice_pdg.py)
# ---------------------------------------------------------------------------

DANGEROUS_SINKS = frozenset({
    "strcpy", "strncpy", "strcat", "strncat",
    "memcpy", "memmove", "memset", "bcopy",
    "sprintf", "snprintf", "vsprintf", "vsnprintf",
    "gets", "fgets", "scanf", "sscanf", "fscanf",
    "read", "recv", "recvfrom", "pread",
    "malloc", "calloc", "realloc", "free",
    "xmalloc", "xrealloc",
    "printf", "fprintf", "syslog", "err", "warn",
})

_COMPARISON_OPS = frozenset({
    "<operator>.lessThan",        "<operator>.lessEqualsThan",
    "<operator>.greaterThan",     "<operator>.greaterEqualsThan",
    "<operator>.equals",          "<operator>.notEquals",
    "<operator>.logicalAnd",      "<operator>.logicalOr",
    "<operator>.logicalNot",
})

_ALLOC_OPS = frozenset({
    "<operator>.alloc", "<operator>.stackAlloc",
    "<operator>.arrayInitializer",
})

# Node types we keep in the graph — everything else is metadata/structural
_KEEP_TYPES = frozenset({
    "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT",
    "IDENTIFIER", "LOCAL", "LITERAL", "FIELD_IDENTIFIER",
    "CALL", "CONTROL_STRUCTURE", "JUMP_TARGET",
    "METHOD_RETURN", "BLOCK",
})

# Edge types from Joern we map to our 3-type scheme
_EDGE_MAP = {
    "CFG":          EDGE_CFG,
    "REACHING_DEF": EDGE_REACHING_DEF,
    "AST":          EDGE_AST,
    "ARGUMENT":     EDGE_AST,   # structural context (same as AST for our purposes)
}


# ---------------------------------------------------------------------------
# Dot file parser
# ---------------------------------------------------------------------------

_QUOTED   = r'"(?:[^"\\]|\\.)*"'
_BODY     = r'(?:[^\]"]*|' + _QUOTED + r')*'
_NODE_RE  = re.compile(r'"(\d+)"\s*\[(' + _BODY + r')\]', re.DOTALL)
_EDGE_RE  = re.compile(r'"(\d+)"\s*->\s*"(\d+)"\s*\[(' + _BODY + r')\]', re.DOTALL)
_ATTR_RE  = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"', re.DOTALL)


def _attrs(s: str) -> dict:
    return dict(_ATTR_RE.findall(s))


def _parse_dot(dot_text: str) -> tuple[dict, list, list]:
    """
    Parse a Joern CPG dot file.

    Returns:
        nodes      — {node_id(int): attr_dict}
        edges      — [(src_id, dst_id, edge_label, property_val)]
        method_ids — IDs of non-external, non-global METHOD nodes
    """
    nodes = {}
    edges = []

    for m in _NODE_RE.finditer(dot_text):
        nid   = int(m.group(1))
        attrs = _attrs(m.group(2))
        nodes[nid] = attrs

    for m in _EDGE_RE.finditer(dot_text):
        src   = int(m.group(1))
        dst   = int(m.group(2))
        attrs = _attrs(m.group(3))
        edges.append((src, dst,
                      attrs.get("label", ""),
                      attrs.get("property", "")))

    method_ids = [
        nid for nid, a in nodes.items()
        if (a.get("label") == "METHOD"
            and a.get("IS_EXTERNAL", "false") != "true"
            and a.get("NAME", "") not in ("<global>", "", "<unknown>"))
    ]

    return nodes, edges, method_ids


# ---------------------------------------------------------------------------
# Node token assignment
# ---------------------------------------------------------------------------

def _canonical_sink(name: str) -> str | None:
    name = name.lstrip("@")
    if name in DANGEROUS_SINKS:
        return name
    # Handle LLVM intrinsic names if they appear
    for s in ("memcpy", "memmove", "memset", "bcopy"):
        if name.startswith(f"llvm.{s}."):
            return s
    return None


def _node_token(attrs: dict) -> int:
    ntype = attrs.get("label", "UNKNOWN")
    if ntype != "CALL":
        return NODE_VOCAB.get(ntype, NODE_VOCAB["UNKNOWN"])
    fn = attrs.get("METHOD_FULL_NAME") or attrs.get("NAME", "")
    if _canonical_sink(fn) is not None:
        return NODE_VOCAB["CALL_SINK"]
    if fn in _COMPARISON_OPS:
        return NODE_VOCAB["CALL_COMPARISON"]
    if fn in _ALLOC_OPS:
        return NODE_VOCAB["CALL_ALLOC"]
    if fn.startswith("<operator>."):
        return NODE_VOCAB["CALL_OPERATOR"]
    return NODE_VOCAB["CALL_REGULAR"]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _collect_fn_nodes(nodes: dict, edges: list, method_id: int) -> set[int]:
    """
    BFS forward over CONTAINS and AST edges from the method node to collect
    all nodes belonging to this function.
    """
    visited = {method_id}
    queue   = deque([method_id])
    while queue:
        nid = queue.popleft()
        for src, dst, elabel, _ in edges:
            if src == nid and elabel in ("CONTAINS", "AST") and dst not in visited:
                visited.add(dst)
                queue.append(dst)
    return visited


def _backward_slice(local_edges: list, sink_ids: set[int],
                    max_hops: int = 15) -> set[int]:
    """
    BFS backward through REACHING_DEF edges from sinks.
    Adds one hop of reverse CFG for control dependence.
    """
    rev_rd  = defaultdict(set)
    rev_cfg = defaultdict(set)
    for src, dst, etype in local_edges:
        if etype == EDGE_REACHING_DEF:
            rev_rd[dst].add(src)
        elif etype == EDGE_CFG:
            rev_cfg[dst].add(src)

    visited = set(sink_ids)
    queue   = deque((sid, 0) for sid in sink_ids)
    while queue:
        nid, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for pred in rev_rd[nid]:
            if pred not in visited:
                visited.add(pred)
                queue.append((pred, depth + 1))
        for pred in rev_cfg[nid]:
            if pred not in visited:
                visited.add(pred)
                queue.append((pred, depth + 1))

    return visited


def cpg_dot_to_graph(dot_text: str, fn_name: str = "") -> dict | None:
    """
    Convert a Joern all-repr dot file to a PDG slice graph dict.

    The output dict is compatible with preprocess_slice_pdg.py output:
      x             (N, 1) int64   — node token IDs
      edge_index    (2, E) int64   — [src; dst]
      edge_type     (E,)   int64   — 0=CFG  1=REACHING_DEF  2=AST
      sink_mask     (N,)   bool    — True at dangerous sink nodes
      sink_fn_names {int: str}     — local_idx → canonical sink name
    """
    nodes, edges, method_ids = _parse_dot(dot_text)
    if not method_ids:
        return None

    method_id = method_ids[0]
    if not fn_name:
        fn_name = nodes[method_id].get("NAME", "unknown")

    # Collect function-local nodes
    fn_node_ids = _collect_fn_nodes(nodes, edges, method_id)

    # Filter to semantically relevant node types
    kept = {nid for nid in fn_node_ids
            if nodes.get(nid, {}).get("label") in _KEEP_TYPES}
    if not kept:
        return None

    id_map     = {nid: i for i, nid in enumerate(sorted(kept))}
    n_total    = len(id_map)

    # Node features
    x             = np.zeros((n_total, 1), dtype=np.int64)
    sink_mask     = np.zeros(n_total, dtype=bool)
    sink_fn_names : dict[int, str] = {}

    for nid, local_idx in id_map.items():
        attrs = nodes[nid]
        tok   = _node_token(attrs)
        x[local_idx, 0] = tok
        if tok == NODE_VOCAB["CALL_SINK"]:
            sink_mask[local_idx] = True
            fn = attrs.get("METHOD_FULL_NAME") or attrs.get("NAME", "")
            sink_fn_names[local_idx] = _canonical_sink(fn) or fn

    # Edges (filter to function-local nodes + recognised edge types)
    local_edges = [
        (id_map[src], id_map[dst], _EDGE_MAP[elabel])
        for src, dst, elabel, _ in edges
        if src in id_map and dst in id_map and elabel in _EDGE_MAP
    ]

    # Backward PDG slice from sinks
    sink_ids = {i for i in range(n_total) if sink_mask[i]}
    sliced   = False

    if sink_ids:
        slice_set  = _backward_slice(local_edges, sink_ids)
        sliced     = len(slice_set) < n_total
        slice_list = sorted(slice_set)
        remap      = {old: new for new, old in enumerate(slice_list)}

        x             = x[slice_list]
        sink_mask     = sink_mask[slice_list]
        sink_fn_names = {remap[k]: v for k, v in sink_fn_names.items() if k in remap}
        local_edges   = [(remap[s], remap[d], et)
                         for s, d, et in local_edges
                         if s in remap and d in remap]

    N = x.shape[0]
    if N == 0:
        return None

    if local_edges:
        srcs = [s for s, d, et in local_edges]
        dsts = [d for s, d, et in local_edges]
        ets  = [et for s, d, et in local_edges]
        edge_index = np.array([srcs, dsts], dtype=np.int64)
        edge_type  = np.array(ets,          dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type  = np.zeros((0,),   dtype=np.int64)

    return {
        "x":             x,
        "edge_index":    edge_index,
        "edge_type":     edge_type,
        "sink_mask":     sink_mask,
        "sink_fn_names": sink_fn_names,
        "_sliced":       sliced,
        "_fn_name":      fn_name,
    }


# ---------------------------------------------------------------------------
# Main API: function source text → graph dict
# ---------------------------------------------------------------------------

def ir_to_graph_joern(func_text: str, fn_name: str = "") -> dict | None:
    """
    Parse a C/C++ function body with Joern and return a PDG slice graph dict.

    Drop-in replacement for ir_to_graph_slice_pdg().  Works without compilation:
    missing struct definitions, undeclared functions, and missing headers are
    handled by Joern's fuzzy island-grammar parser.

    Args:
        func_text : raw C/C++ function source text
        fn_name   : optional function name hint (inferred from CPG if empty)

    Returns:
        graph dict with keys: x, edge_index, edge_type, sink_mask, sink_fn_names
        None if Joern cannot parse the input or no nodes are extracted
    """
    if not JOERN_PARSE.exists():
        raise RuntimeError(
            f"joern-parse not found at {JOERN_PARSE}\n"
            f"Install Joern or set JOERN_HOME env var."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp      = Path(tmpdir)
        src_file = tmp / "func.c"
        cpg_file = tmp / "cpg.bin"
        dot_dir  = tmp / "dot"

        src_file.write_text(func_text, encoding="utf-8", errors="replace")

        r = subprocess.run(
            [str(JOERN_PARSE), str(src_file), "--output", str(cpg_file)],
            capture_output=True, timeout=60,
        )
        if r.returncode != 0 or not cpg_file.exists():
            return None

        r = subprocess.run(
            [str(JOERN_EXPORT), "--repr=all", "--format=dot",
             "--out", str(dot_dir), str(cpg_file)],
            capture_output=True, timeout=60,
        )
        if r.returncode != 0:
            return None

        dot_files = list(dot_dir.glob("*.dot"))
        if not dot_files:
            return None

        dot_text = dot_files[0].read_text(encoding="utf-8", errors="replace")
        return cpg_dot_to_graph(dot_text, fn_name=fn_name)


# ---------------------------------------------------------------------------
# Standalone CLI — smoke test
# ---------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("source", help=".c / .cpp source file")
    ap.add_argument("--json", action="store_true", help="print JSON summary")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"ERROR: {src} not found"); sys.exit(1)

    g = ir_to_graph_joern(src.read_text(errors="replace"))
    if g is None:
        print("ERROR: graph extraction failed"); sys.exit(1)

    N = g["x"].shape[0]
    E = g["edge_index"].shape[1]
    inv = {v: k for k, v in NODE_VOCAB.items()}
    from collections import Counter
    counts = Counter(int(g["x"][i, 0]) for i in range(N))

    print(f"\nFunction : {g['_fn_name']}")
    print(f"Nodes    : {N}   Edges: {E}   Sliced: {g['_sliced']}")
    print(f"Sinks    : {len(g['sink_fn_names'])}  {list(g['sink_fn_names'].values())}")
    guards = counts.get(NODE_VOCAB["CALL_COMPARISON"], 0)
    print(f"Guards   : {guards} comparison node(s)")
    print(f"\nNode breakdown:")
    for tok, cnt in sorted(counts.items()):
        print(f"  {inv.get(tok,'?'):25s} {cnt:4d}")

    if args.json:
        print(json.dumps({
            "fn_name":   g["_fn_name"],
            "n_nodes":   N,
            "n_edges":   E,
            "n_sinks":   len(g["sink_fn_names"]),
            "n_guards":  guards,
            "sliced":    g["_sliced"],
            "sinks":     {str(k): v for k, v in g["sink_fn_names"].items()},
        }, indent=2))


if __name__ == "__main__":
    _cli()
