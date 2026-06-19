#!/usr/bin/env python3
"""
diagnose_attrition.py — Sample PrimeVul failures to find what compile_to_ir
                         cannot fix, and how many are C++ vs C.

Run from train_gnn/:
    python diagnose_attrition.py                        # 300 random functions
    python diagnose_attrition.py --n 1000               # larger sample
    python diagnose_attrition.py --cache /path/to/raw   # explicit cache path
"""

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

try:
    from preprocess import PREAMBLE, _try_compile, _strip_asm_blocks
except ImportError:
    print("ERROR: run this script from the train_gnn/ directory")
    sys.exit(1)

# ── helpers ──────────────────────────────────────────────────────────────────

_CPP_SIGNALS = re.compile(
    r"\bclass\b|\btemplate\b|\bnamespace\b|\bpublic:\b|\bprivate:\b|"
    r"::|std::|nullptr\b|\bnew\b\s+\w|\bdelete\b\s+[\w\[]|"
    r"<\w+(?:,\s*\w+)*>",
    re.MULTILINE,
)

def _is_cpp(source: str) -> bool:
    return bool(_CPP_SIGNALS.search(source))


def _first_error_category(stderr: str) -> str:
    """Collapse clang stderr to a single category string."""
    patterns = [
        (r"expected unqualified-id",              "syntax/cpp-keyword"),
        (r"use of undeclared identifier 'nullptr'","cpp-nullptr"),
        (r"template\b",                            "cpp-template"),
        (r"'class' keyword",                       "cpp-class"),
        (r"ambiguous",                             "cpp-overload"),
        (r"cannot initialize a variable of type .* with an rvalue of type", "cpp-type"),
        (r"no member named '(\w+)' in",            "no-member"),
        (r"unknown type name '(\w+)'",             "unknown-type"),
        (r"use of undeclared identifier",          "undecl-ident"),
        (r"implicit declaration of function",      "implicit-func"),
        (r"incomplete type",                       "incomplete-type"),
        (r"cannot combine with previous",          "combine-error"),
        (r"expression is not assignable",          "not-assignable"),
        (r"too many arguments",                    "too-many-args"),
        (r"conflicting types",                     "conflicting-types"),
        (r"redefinition",                          "redefinition"),
        (r"initializer element is not a compile",  "non-const-init"),
        (r"expected ';'",                          "syntax/missing-semi"),
        (r"expected '\)'",                         "syntax/missing-paren"),
        (r"expected expression",                   "syntax/expected-expr"),
        (r"error: ",                               "other-error"),
    ]
    for pat, label in patterns:
        if re.search(pat, stderr, re.IGNORECASE):
            return label
    return "unknown"


def _try_compile_cpp(source: str) -> tuple[str | None, str]:
    """Try compiling as C++ with a reduced preamble."""
    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
        f.write(source)
        fname = f.name
    try:
        r = subprocess.run(
            ["clang++", "-O0", "-fno-inline", "-S", "-emit-llvm",
             "-w", "-ferror-limit=5", "-x", "c++", fname, "-o", "/dev/null"],
            capture_output=True, text=True, timeout=30,
        )
        return (fname if r.returncode == 0 else None), r.stderr
    except Exception:
        return None, ""
    finally:
        Path(fname).unlink(missing_ok=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n",     type=int, default=300, help="Functions to sample")
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--cache", type=str, default=None,
                    help="Path to primevul_raw.jsonl (default: data/primevul_raw.jsonl)")
    ap.add_argument("--cpp-retry", action="store_true",
                    help="For C-compile failures with C++ signals, retry as C++")
    args = ap.parse_args()

    cache = Path(args.cache) if args.cache else HERE / "data" / "primevul_raw.jsonl"
    if not cache.exists():
        print(f"ERROR: {cache} not found"); sys.exit(1)

    print(f"Loading {cache} ...")
    with open(cache) as f:
        items = [json.loads(l) for l in f]
    print(f"  {len(items)} total functions")

    rng = random.Random(args.seed)
    sample = rng.sample(items, min(args.n, len(items)))

    ok = fail = cpp_detected = cpp_recovered = 0
    error_counter: Counter = Counter()
    cpp_error_counter: Counter = Counter()
    first_errors: list[str] = []          # raw stderr first line for examples

    print(f"\nTrying {len(sample)} functions with compile_to_ir (max_retries=20)...")

    for i, item in enumerate(sample, 1):
        src = _strip_asm_blocks(item["func"])
        ir, stderr = _try_compile(PREAMBLE + "\n" + src)

        # compile_to_ir does full retry; here we just do a single attempt
        # to get the raw first-error distribution. For a closer match to
        # production, import and call compile_to_ir directly:
        try:
            from preprocess import compile_to_ir
            ir_full = compile_to_ir(item["func"])
        except Exception:
            ir_full = None

        if ir_full is not None:
            ok += 1
        else:
            fail += 1
            # get fresh first-error for categorisation
            _, err = _try_compile(PREAMBLE + "\n" + src)
            cat = _first_error_category(err)
            is_cpp = _is_cpp(src)

            if is_cpp:
                cpp_detected += 1
                cpp_error_counter[cat] += 1
                if args.cpp_retry:
                    ir2, _ = _try_compile_cpp(src)
                    if ir2:
                        cpp_recovered += 1
            else:
                error_counter[cat] += 1

            if len(first_errors) < 5 and err.strip():
                lines = [l for l in err.splitlines() if "error:" in l]
                first_errors.append(lines[0] if lines else err.splitlines()[0])

        if i % 50 == 0:
            print(f"  {i}/{len(sample)}  ok={ok}  fail={fail}", flush=True)

    total = ok + fail
    print(f"\n{'─'*60}")
    print(f"Sample size  : {total}")
    print(f"Compiled OK  : {ok}  ({100*ok/total:.1f}%)")
    print(f"Failed       : {fail}  ({100*fail/total:.1f}%)")
    if fail:
        print(f"\nC++ signals detected in failures : {cpp_detected} "
              f"({100*cpp_detected/fail:.1f}% of failures)")
        if args.cpp_retry and cpp_detected:
            print(f"C++ recovered by clang++         : {cpp_recovered} "
                  f"({100*cpp_recovered/max(cpp_detected,1):.1f}% of C++ failures)")

        print(f"\n── Error categories (C functions) ──")
        for cat, cnt in error_counter.most_common(15):
            print(f"  {cat:35s}  {cnt:4d}  ({100*cnt/fail:.1f}%)")

        if cpp_detected:
            print(f"\n── Error categories (C++ functions) ──")
            for cat, cnt in cpp_error_counter.most_common(10):
                print(f"  {cat:35s}  {cnt:4d}")

        if first_errors:
            print(f"\n── Sample first-error lines ──")
            for e in first_errors:
                print(f"  {e[:120]}")

    print(f"\nConclusion: to improve beyond {100*ok/total:.0f}%, look at the top "
          f"category above and decide if it's fixable.")


if __name__ == "__main__":
    main()
