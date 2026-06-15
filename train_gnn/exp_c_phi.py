#!/usr/bin/env python3
"""
Experiment C — Phi node operand format in llvmlite.

Tests whether op.name on phi node operands cleanly exposes SSA value names
and block label names, so the existing DFG edge filter (skip block_names)
handles phi nodes correctly without a special branch.

Requires llvmlite:  pip install llvmlite

Usage:
    python exp_c_phi.py
"""

import llvmlite.binding as llvm

IR = """
define i32 @sum_to_n(i32 %n) {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum.0 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %add = add i32 %sum.0, %i.0
  %inc = add i32 %i.0, 1
  br label %for.cond

for.end:
  ret i32 %sum.0
}
"""


def main():
    mod = llvm.parse_assembly(IR)
    func = next(iter(mod.functions))
    block_names = {block.name for block in func.blocks}

    print(f"Block names (will be filtered from DFG): {block_names}\n")
    print("Full instruction walk with operand details:\n")

    dfg_edges = []
    instr_id: dict[str, int] = {}
    gid = 0

    for block in func.blocks:
        print(f"  [{block.name}]")
        for instr in block.instructions:
            instr_id[instr.name] = gid
            ops = [(op.name, str(op.type)) for op in instr.operands]
            print(f"    {gid:2d}  {instr.opcode:12}  name={instr.name!r:8}  operands={ops}")
            gid += 1
        print()

    print("Simulating DFG edge extraction (skip block labels and empty names):\n")
    gid = 0
    for block in func.blocks:
        for instr in block.instructions:
            for op in instr.operands:
                if op.name and op.name not in block_names:
                    src = instr_id.get(op.name)
                    if src is not None:
                        dfg_edges.append((src, gid))
                        print(f"  DFG edge: {src} ({op.name}) → {gid} ({instr.name})")
            gid += 1

    print(f"\nDFG edges: {dfg_edges}")
    print(f"\nKey question: do phi operands appear in the list above correctly?")
    print("Expected: %add and %inc should generate edges INTO for.cond phi nodes.")
    print("          %entry, %for.body labels should be silently skipped.")


if __name__ == "__main__":
    main()
