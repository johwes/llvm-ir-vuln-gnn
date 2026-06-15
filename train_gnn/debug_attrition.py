#!/usr/bin/env python3
"""
debug_attrition.py — diagnose 100% attrition in preprocess_slice_pdg.py

Run from the train_gnn directory:
  python3 debug_attrition.py

Prints which step is failing and the clang error message.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from preprocess import compile_to_ir, _try_compile, PREAMBLE

# Test 1: can clang compile at all?
simple = 'int add(int a, int b) { return a + b; }'
ir, stderr = _try_compile(PREAMBLE + '\n' + simple)
print("=" * 60)
print(f"Test 1 — simple `int add(int,int)`:  {'PASS' if ir else 'FAIL'}")
if not ir:
    print(f"  clang stderr: {stderr[:600]}")

ir2 = compile_to_ir(simple)
print(f"Test 2 — compile_to_ir(simple):      {'PASS' if ir2 else 'FAIL'}")

# Test 3: try first item in train.jsonl
jsonl = HERE / "data" / "train.jsonl"
if jsonl.exists():
    with open(jsonl) as f:
        item = json.loads(f.readline())
    ir3, stderr3 = _try_compile(PREAMBLE + '\n' + item["func"])
    print(f"Test 3 — first train item (1-shot):  {'PASS' if ir3 else 'FAIL'}")
    if not ir3:
        print(f"  func[:120]: {repr(item['func'][:120])}")
        print(f"  clang stderr: {stderr3[:600]}")

    # How many of first 20 train items compile?
    ok = fail = 0
    with open(jsonl) as f:
        for i, line in enumerate(f):
            if i >= 20: break
            it = json.loads(line)
            r = compile_to_ir(it["func"])
            if r: ok += 1
            else: fail += 1
    print(f"Test 4 — compile_to_ir on first 20:  {ok} OK / {fail} FAIL")
else:
    print("Test 3/4 — data/train.jsonl not found, skipping")

# Test 5: check that ir_to_graph_slice_pdg works on a trivial IR
try:
    from preprocess_slice_pdg import ir_to_graph_slice_pdg
    trivial_c = 'void f(char *dst, char *src) { __builtin_memcpy(dst, src, 4); }'
    ir5 = compile_to_ir(trivial_c)
    if ir5:
        g = ir_to_graph_slice_pdg(ir5)
        print(f"Test 5 — ir_to_graph_slice_pdg:      {'PASS' if g else 'FAIL (returned None)'}")
        if g:
            print(f"  nodes={g['x'].shape}, edges={g['edge_index'].shape}")
    else:
        print("Test 5 — skipped (compile failed)")
except Exception as e:
    print(f"Test 5 — EXCEPTION: {type(e).__name__}: {e}")

print("=" * 60)
