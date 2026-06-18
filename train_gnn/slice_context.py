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

    # ---- natural language + harness hint --------------------------------
    sink_strs  = []
    hint_parts = []

    for sink in sinks:
        fn = sink.get("fn") or "unknown"
        info = _SINK_INFO.get(fn)
        if info:
            what, probe = info
            sink_strs.append(f"`{fn}` ({what})")
            hint_parts.append(f"fuzz {probe}")
        else:
            sink_strs.append(f"`{fn}` (dangerous operation)")
            hint_parts.append("fuzz all arguments")

    if not sinks:
        sink_strs  = ["no explicit dangerous sink identified — scored by full-graph structure"]
        hint_parts = ["fuzz all inputs broadly"]

    channel_note = " or ".join(input_channels)
    guard_note   = (
        f"{guard_count} comparison(s) in slice — verify they guard the sink path"
        if has_guard
        else "no comparison (icmp) in slice — sink appears UNGUARDED"
    )

    natural_language = (
        f"Function `{fn_name}` contains: {'; '.join(sink_strs)}. "
        f"Input originates from: {channel_note}. "
        f"Guard status: {guard_note}. "
        f"Slice: {N} nodes, {len(sinks)} dangerous sink(s)."
    )

    harness_hint = " | ".join(hint_parts)

    return {
        "fn_name":          fn_name,
        "slice_size":       N,
        "n_sinks":          len(sinks),
        "sinks":            sinks,
        "input_channels":   input_channels,
        "guard_count":      guard_count,
        "has_guard":        has_guard,
        "natural_language": natural_language,
        "harness_hint":     harness_hint,
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

    sink_labels = []
    for s in summary["sinks"]:
        fn = s.get("fn", "unknown")
        info = _SINK_INFO.get(fn)
        short = info[0][:55] if info else "dangerous operation"
        sink_labels.append(f"{fn} — {short}")
    lines.append("Sinks           : " + ("; ".join(sink_labels) if sink_labels
                                          else "none identified"))

    lines.append("Input channels  : " + ", ".join(summary["input_channels"]))

    guard = (f"{summary['guard_count']} comparison(s) present — verify they guard sink"
             if summary["has_guard"]
             else "NO icmp in slice — sink appears UNGUARDED")
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

    # Split into per-function IR segments, preserving module-level preamble
    # (declares, globals) so that dangerous sink detection works per-function.
    _NAME_RE = re.compile(r'@([\w.$]+)\s*\(')
    def _split_functions(text):
        segs = re.split(r'(?=^define\b)', text, flags=re.MULTILINE)
        preamble = segs[0] if segs and not segs[0].strip().startswith("define") else ""
        out = []
        for seg in segs:
            seg = seg.strip()
            if not seg.startswith("define"):
                continue
            m = _NAME_RE.search(seg[:300])
            if m:
                # Prepend module preamble so declares/globals are visible
                out.append((m.group(1), preamble + "\n" + seg))
        return out

    functions = _split_functions(ir_text)
    if not functions:
        # Single-function file — try directly
        m = re.search(r'@([\w.$]+)\s*\(', ir_text[:500])
        fn_name = m.group(1) if m else Path(args.ir_file).stem
        functions = [(fn_name, ir_text)]

    for fn_name, fn_ir in functions:
        if args.debug:
            try:
                import llvmlite.binding as _llvm
                _llvm.parse_assembly(fn_ir)
                print(f"[{fn_name}] parse OK")
            except Exception as exc:
                print(f"[{fn_name}] parse FAILED: {exc}")
                continue
        g = ir_to_graph_slice_pdg(fn_ir)
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
