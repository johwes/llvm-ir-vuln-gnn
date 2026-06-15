#!/usr/bin/env python3
"""
debug_predicate.py — Verify that llvmlite exposes icmp predicates via str(instr).

Tests two C snippets that differ only in comparison operator (sgt vs sge).
Prints what the current vocabulary assigns vs what enriched vocab would assign.

Usage:
    python debug_predicate.py
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import llvmlite.binding as llvm
except ImportError:
    sys.exit("llvmlite not installed. Run: pip install llvmlite")

# --- Snippets that differ only in comparison operator ---
VULN_C = """\
int check(int index, int size) {
    if (index > size) return -1;
    return 0;
}
"""
FIXED_C = """\
int check(int index, int size) {
    if (index >= size) return -1;
    return 0;
}
"""

ICMP_PRED_RE = re.compile(r'\bicmp\s+(\w+)\b')

# Current vocab (everything collapses to 46)
CURRENT_VOCAB = {"icmp": 46}

# Enriched vocab (each predicate gets its own ID)
ICMP_PRED_VOCAB = {
    "icmp_eq":  46, "icmp_ne":  47,
    "icmp_slt": 48, "icmp_sle": 49,
    "icmp_sgt": 50, "icmp_sge": 51,
    "icmp_ult": 52, "icmp_ule": 53,
    "icmp_ugt": 54, "icmp_uge": 55,
}


def compile_to_ir(src: str) -> str | None:
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run(
            ["clang", "-O0", "-fno-inline", "-S", "-emit-llvm", "-o", "-", path],
            capture_output=True, text=True,
        )
        return r.stdout if r.returncode == 0 else None
    finally:
        Path(path).unlink(missing_ok=True)


def scan_icmp(ir_text: str, label: str):
    mod = llvm.parse_assembly(ir_text)
    print(f"\n=== {label} ===")
    for fn in mod.functions:
        if fn.is_declaration:
            continue
        for bb in fn.blocks:
            for instr in bb.instructions:
                if instr.opcode != "icmp":
                    continue
                instr_str = str(instr).strip()
                print(f"  raw IR:        {instr_str!r}")
                m = ICMP_PRED_RE.search(instr_str)
                pred = m.group(1) if m else "??"
                print(f"  predicate:     {pred!r}")
                current_id  = CURRENT_VOCAB.get("icmp", 79)
                enriched_id = ICMP_PRED_VOCAB.get(f"icmp_{pred}", 79)
                print(f"  current  ID:   {current_id}  (icmp → always 46)")
                print(f"  enriched ID:   {enriched_id}  (icmp_{pred})")
                print()


def main():
    print("Compiling snippets with clang -O0 -fno-inline ...")
    vuln_ir  = compile_to_ir(VULN_C)
    fixed_ir = compile_to_ir(FIXED_C)
    if not vuln_ir or not fixed_ir:
        sys.exit("clang compilation failed — is clang in PATH?")

    scan_icmp(vuln_ir,  "VULNERABLE  (index > size   →  icmp sgt)")
    scan_icmp(fixed_ir, "FIXED       (index >= size  →  icmp sge)")

    print("=" * 60)
    print("VERDICT:")
    print("  Current vocab  → both nodes get ID 46 (indistinguishable)")
    print("  Enriched vocab → sgt=50, sge=51 (model can see the patch)")
    print()
    print("Next step: update OPCODE_VOCAB in preprocess_instr.py")
    print("  Replace 'icmp': 46 with 10 predicate-specific entries")
    print("  Bump VOCAB_SIZE from 80 to ~100")
    print("  Re-run preprocess_instr.py + train_instr.py (§10 experiment)")


if __name__ == "__main__":
    main()
