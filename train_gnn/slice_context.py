#!/usr/bin/env python3
"""
slice_context.py — Convert a PDG slice graph into structured vulnerability context.

Takes a slice dict produced by preprocess_slice_pdg.py (or v3) and returns
a structured summary of the vulnerability pattern: what the dangerous sink is,
where the input comes from, whether a guard is present, and natural-language
context ready for injection into an LLM harness-generation prompt.

The goal is to pre-compute the hard analysis step (what is dangerous and why)
so that a small sparse model (e.g. Qwen 35B-A3B) only has to do the easy step
(generate code from a specification) rather than reason about data flow itself.

Usage (standalone):
    python slice_context.py path/to/function.ll
    python slice_context.py path/to/function.ll --json

Usage (as library):
    from slice_context import summarize_slice, format_for_llm
    summary = summarize_slice(graph_dict, fn_name="process_packet")
    prompt_block = format_for_llm(summary, score=0.82)
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Opcode constants (must match OPCODE_VOCAB in preprocess_slice_pdg.py)
# ---------------------------------------------------------------------------

IDX_CONTEXT   = 0
IDX_ARGUMENT  = 1
IDX_MOCK      = 75    # external function / global variable reference
IDX_CONST_INT = 76
IDX_CONST_FP  = 77
IDX_UNDEF     = 78
IDX_UNKNOWN   = 79

ICMP_OPCODES = frozenset({
    46,              # icmp (unspecified predicate)
    80, 81,          # eq, ne
    82, 83, 84, 85,  # slt, sle, sgt, sge
    86, 87, 88, 89,  # ult, ule, ugt, uge
})

# Human-readable display names for IR opcodes that appear as sinks.
# Keyed by the internal name used in _SINK_INFO / sink_fn_names.
_DISPLAY_NAMES: dict[str, str] = {
    "getelementptr": "array/ptr-subscript",
}

# Dangerous sink descriptions: fn_name → (what it does, what to fuzz)
_SINK_INFO: dict[str, tuple[str, str]] = {
    "strcpy":   ("copies string to dest without bounds check",
                 "source string length vs dest buffer size"),
    "strncpy":  ("copies at most n bytes — check dest capacity vs n",
                 "n relative to dest size; source not null-terminated when n < len(src)"),
    "strcat":   ("appends string without bounds check",
                 "remaining dest capacity vs source length"),
    "strncat":  ("appends at most n bytes",
                 "n relative to remaining dest capacity"),
    "memcpy":   ("copies n bytes from src to dest — no overlap or bounds check",
                 "n relative to dest buffer size; n=0, n=SIZE_MAX, n=dest_size+1"),
    "memmove":  ("moves n bytes — allows overlap, no bounds check",
                 "n relative to dest size"),
    "memset":   ("fills n bytes — no bounds check",
                 "n relative to buffer size"),
    "bcopy":    ("copies n bytes src→dest (deprecated) — no bounds check",
                 "n relative to dest size"),
    "sprintf":  ("formats into dest without length limit",
                 "formatted output length vs dest size; user-controlled format string"),
    "snprintf": ("formats at most n bytes — verify n matches dest",
                 "n relative to dest size; off-by-one at n"),
    "vsprintf": ("formats variadic args without length limit",
                 "formatted output length vs dest size"),
    "vsnprintf":("formats variadic args, at most n bytes",
                 "n relative to dest size"),
    "gets":     ("reads line with no length limit — inherently unsafe",
                 "any non-empty input; function is unconditionally vulnerable"),
    "fgets":    ("reads at most n-1 chars — check n matches buffer",
                 "n relative to buffer size; input exactly n-1 chars with no newline"),
    "scanf":    ("reads formatted input — %%s without width is unbounded",
                 "input longer than any %%s field width; format string control"),
    "sscanf":   ("reads from string — %%s without width is unbounded",
                 "input longer than any %%s field width"),
    "fscanf":   ("reads from file — %%s without width is unbounded",
                 "input longer than any %%s field width"),
    "read":     ("reads up to count bytes — no null termination",
                 "count relative to buffer size; count=0; count=SIZE_MAX"),
    "recv":     ("receives up to len bytes — no null termination",
                 "len relative to buffer size; len=0; len=INT_MAX"),
    "recvfrom": ("receives up to len bytes from socket",
                 "len relative to buffer size; truncated messages"),
    "pread":    ("reads up to count bytes at offset",
                 "count relative to buffer size; offset overflow"),
    "malloc":   ("allocates heap — return may be null; size may overflow",
                 "size=0; size=SIZE_MAX; integer overflow in size computation"),
    "calloc":   ("allocates zeroed heap — n*size may overflow; return may be null",
                 "n*size integer overflow; n=0 or size=0; very large n"),
    "realloc":  ("reallocates — null return leaves original pointer valid but lost",
                 "new_size=0 (frees); new_size=SIZE_MAX; null return before use"),
    "free":     ("frees heap memory — double-free or use-after-free if called again",
                 "call target twice with same pointer; access after free"),
    "xmalloc":  ("malloc wrapper — check whether it aborts or returns null on failure",
                 "very large size; size=0"),
    "xrealloc": ("realloc wrapper",
                 "new_size=0; very large size"),
    "printf":   ("format string to stdout — dangerous if format is user-controlled",
                 "user-controlled format string with %%n, %%s, %%x"),
    "fprintf":  ("format string to file",
                 "user-controlled format string"),
    "syslog":   ("format string to syslog",
                 "user-controlled format string with %%n"),
    "err":      ("format string + perror to stderr",
                 "user-controlled format string"),
    "warn":     ("format string warning to stderr",
                 "user-controlled format string"),
    "getelementptr": ("pointer arithmetic with non-constant index — out-of-bounds if unchecked",
                      "index at, near, and beyond array bounds; negative index; index=SIZE_MAX"),
}


# ---------------------------------------------------------------------------
# Core summarizer
# ---------------------------------------------------------------------------

def summarize_slice(g: dict, fn_name: str = "unknown") -> dict:
    """
    Convert a PDG slice dict into structured vulnerability context.

    Slice dict keys (preprocess_slice_pdg.py format):
      x             (N, 1) int64  — opcode ID per node
      edge_index    (2, E) int64  — [src; dst]
      edge_type     (E,)   int64  — 0=CFG, 1=DFG, 2=context
      sink_mask     (N,)   bool   — True at sink nodes (v3 only; inferred if absent)
      sink_fn_names {int: str}    — node_idx → dangerous function name at that sink

    Returns dict:
      fn_name, slice_size, n_sinks, sinks (list),
      input_channels (list), guard_count, has_guard,
      natural_language (str), harness_hint (str)
    """
    x          = g["x"]
    N          = x.shape[0]
    edge_index = g["edge_index"]
    edge_type  = g["edge_type"]
    E          = edge_index.shape[1] if (edge_index.ndim == 2 and edge_index.size > 0) else 0

    opcodes = [int(x[i, 0]) for i in range(N)]

    sink_fn_names: dict[int, str] = g.get("sink_fn_names", {})
    sink_mask = g.get("sink_mask", None)

    # ---- identify sinks ------------------------------------------------
    sinks = []

    if sink_mask is not None:
        for i in range(N):
            if not sink_mask[i]:
                continue
            opc = opcodes[i]
            if opc == 63:   # call
                fn = sink_fn_names.get(i, "unknown")
                sinks.append({"node": i, "type": "dangerous_call", "fn": fn})
            elif opc == 29: # getelementptr
                sinks.append({"node": i, "type": "unguarded_gep", "fn": "getelementptr"})
    else:
        # Fallback: scan for call and GEP nodes when sink_mask absent
        for i in range(N):
            opc = opcodes[i]
            if opc == 63 and i in sink_fn_names:
                sinks.append({"node": i, "type": "dangerous_call",
                               "fn": sink_fn_names[i]})
            elif opc == 29:
                # Only flag GEP if it has a non-constant DFG predecessor
                preds = []
                for e in range(E):
                    if int(edge_type[e]) == 1 and int(edge_index[1, e]) == i:
                        preds.append(int(edge_index[0, e]))
                const_ids = {IDX_CONST_INT, IDX_CONST_FP, IDX_UNDEF, IDX_CONTEXT}
                if any(opcodes[p] not in const_ids for p in preds):
                    sinks.append({"node": i, "type": "unguarded_gep",
                                  "fn": "getelementptr"})

    # ---- input channels ------------------------------------------------
    input_channels = []
    if any(opc == IDX_ARGUMENT for opc in opcodes):
        input_channels.append("function_argument")
    if any(opc == IDX_MOCK for opc in opcodes):
        input_channels.append("external_call_return")
    if not input_channels:
        input_channels.append("internal_computation")

    # ---- guard check ---------------------------------------------------
    guard_count = sum(1 for opc in opcodes if opc in ICMP_OPCODES)
    has_guard   = guard_count > 0

    # ---- deduplicate sinks by function name (preserve first-seen order) ----
    from collections import Counter, OrderedDict
    sink_counts: dict[str, int] = Counter(s.get("fn") or "unknown" for s in sinks)
    seen: set[str] = set()
    unique_sinks = []
    for s in sinks:
        fn = s.get("fn") or "unknown"
        if fn not in seen:
            seen.add(fn)
            unique_sinks.append(s)

    # ---- natural language + harness hint (deduplicated) -----------------
    sink_strs  = []
    hint_parts = []

    for s in unique_sinks:
        fn      = s.get("fn") or "unknown"
        display = _DISPLAY_NAMES.get(fn, fn)
        count   = sink_counts[fn]
        info    = _SINK_INFO.get(fn)
        suffix  = f" ×{count}" if count > 1 else ""
        if info:
            what, probe = info
            sink_strs.append(f"`{display}`{suffix} ({what})")
            if f"fuzz {probe}" not in hint_parts:
                hint_parts.append(f"fuzz {probe}")
        else:
            sink_strs.append(f"`{display}`{suffix} (dangerous operation)")
            if "fuzz all arguments" not in hint_parts:
                hint_parts.append("fuzz all arguments")

    if not sinks:
        sink_strs  = ["no explicit dangerous sink identified — scored by full-graph structure"]
        hint_parts = ["fuzz all inputs broadly"]

    # ---- guard density -------------------------------------------------
    n_sinks = len(sinks)
    if n_sinks == 0:
        guard_density       = 0.0
        guard_density_label = "no sinks"
    elif not has_guard:
        guard_density       = float("inf")
        guard_density_label = "UNGUARDED"
    else:
        guard_density = n_sinks / guard_count   # sinks per guard — higher = worse
        if guard_density >= 10:
            guard_density_label = "very sparse"
        elif guard_density >= 5:
            guard_density_label = "sparse"
        elif guard_density >= 2:
            guard_density_label = "moderate"
        else:
            guard_density_label = "well-covered"

    channel_note = " or ".join(input_channels)
    if not has_guard:
        guard_note = "no comparison (icmp) in slice — sink appears UNGUARDED"
    elif n_sinks == 0:
        guard_note = f"{guard_count} comparison(s) in slice"
    else:
        guard_note = (
            f"{guard_count} guard(s) / {n_sinks} sink(s)"
            f" = {guard_density:.1f} sinks/guard ({guard_density_label})"
        )

    natural_language = (
        f"Function `{fn_name}` contains: {'; '.join(sink_strs)}. "
        f"Input originates from: {channel_note}. "
        f"Guard status: {guard_note}. "
        f"Slice: {N} nodes, {n_sinks} sink(s) ({len(unique_sinks)} unique type(s))."
    )

    harness_hint = " | ".join(hint_parts)

    return {
        "fn_name":            fn_name,
        "slice_size":         N,
        "n_sinks":            n_sinks,
        "n_unique_sinks":     len(unique_sinks),
        "sinks":              sinks,
        "sink_counts":        dict(sink_counts),
        "input_channels":     input_channels,
        "guard_count":        guard_count,
        "has_guard":          has_guard,
        "guard_density":      guard_density,
        "guard_density_label":guard_density_label,
        "natural_language":   natural_language,
        "harness_hint":       harness_hint,
    }


def format_for_llm(summary: dict, score: float | None = None,
                   width: int = 60) -> str:
    """
    Format a slice summary as a labelled block for LLM prompt injection.

    Example output:
        === GNN Vulnerability Context ===
        Suspicion score : 73.2%  (SUSPICIOUS)
        Sinks           : memcpy (copies n bytes without bounds check)
        Input channels  : function_argument
        Guard status    : no comparison in slice — sink UNGUARDED
        Harness target  : fuzz n relative to dest buffer size; n=0, n=SIZE_MAX
        Slice           : 31 nodes, 1 sink
        ==================================
    """
    sep = "=" * width
    lines = [sep, "GNN Vulnerability Context"]

    if score is not None:
        flag = "SUSPICIOUS" if score >= 0.5 else "low suspicion"
        lines.append(f"Suspicion score : {score:.1%}  ({flag})")

    # Deduplicate: one label per unique sink type, with ×N count
    sink_counts = summary.get("sink_counts", {})
    seen: set[str] = set()
    sink_labels = []
    for s in summary["sinks"]:
        fn = s.get("fn", "unknown")
        if fn in seen:
            continue
        seen.add(fn)
        display = _DISPLAY_NAMES.get(fn, fn)
        info    = _SINK_INFO.get(fn)
        short   = info[0][:50] if info else "dangerous operation"
        count   = sink_counts.get(fn, 1)
        tag     = f" ×{count}" if count > 1 else ""
        sink_labels.append(f"{display}{tag} — {short}")
    lines.append("Sinks           : " + ("; ".join(sink_labels) if sink_labels
                                          else "none identified"))

    lines.append("Input channels  : " + ", ".join(summary["input_channels"]))

    n_sinks = summary["n_sinks"]
    gc      = summary["guard_count"]
    if not summary["has_guard"]:
        guard = "NO icmp in slice — sink appears UNGUARDED"
    elif n_sinks == 0:
        guard = f"{gc} comparison(s) present"
    else:
        gd    = summary.get("guard_density", n_sinks / gc)
        label = summary.get("guard_density_label", "")
        guard = f"{gc} guard(s) / {n_sinks} sink(s) = {gd:.1f} sinks/guard ({label})"
    lines.append("Guard status    : " + guard)
    lines.append("Harness target  : " + summary["harness_hint"])
    lines.append(f"Slice           : {summary['slice_size']} nodes, "
                 f"{summary['n_sinks']} sink(s)")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone CLI (demo mode — parses IR directly without a trained model)
# ---------------------------------------------------------------------------

def _demo_cli():
    import argparse
    import json as _json
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir_file", help=".ll file to analyse")
    ap.add_argument("--json",  action="store_true", help="output raw JSON summary")
    ap.add_argument("--debug", action="store_true", help="show parse errors instead of silencing them")
    args = ap.parse_args()

    HERE = Path(__file__).parent
    sys.path.insert(0, str(HERE))
    from preprocess_slice_pdg import ir_to_graph_slice_pdg

    import re
    ir_text = Path(args.ir_file).read_text(errors="replace")

    # Parse the full module once to enumerate non-declaration functions.
    # Passing the full IR (not a per-function split) to ir_to_graph_slice_pdg
    # preserves all declare stubs and globals — cross-function calls that appear
    # in the same source file are visible, and llvmlite won't reject them.
    import llvmlite.binding as _llvm
    try:
        _mod = _llvm.parse_assembly(ir_text)
    except Exception as exc:
        if args.debug:
            print(f"Module parse FAILED: {exc}")
        print(f"ERROR: could not parse {args.ir_file}: {exc}")
        sys.exit(1)

    functions = [(fn.name, fn.name)
                 for fn in _mod.functions if not fn.is_declaration]

    if not functions:
        print(f"ERROR: no non-declaration functions found in {args.ir_file}")
        sys.exit(1)

    for fn_name, _ in functions:
        if args.debug:
            print(f"[{fn_name}] parse OK (full-module mode)")
        g = ir_to_graph_slice_pdg(ir_text, fn_name=fn_name)
        if g is None:
            print(f"[{fn_name}] — could not extract graph (no basic blocks)\n")
            continue

        summary = summarize_slice(g, fn_name=fn_name)

        if args.json:
            print(_json.dumps({k: v for k, v in summary.items()
                                if k != "sinks"}, indent=2))
        else:
            print(format_for_llm(summary))
            print(f"Natural language:\n  {summary['natural_language']}\n")


if __name__ == "__main__":
    _demo_cli()
