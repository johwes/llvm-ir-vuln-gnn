#!/bin/bash
# Compile sample C files to LLVM IR and run the structural embedding demo.
#
# Requirements: clang (any version), python3
# No GPU, no pip installs, no dependencies beyond the standard library.
#
# Usage:
#   ./run.sh              — use local samples/ directory
#   ./run.sh --from-repo  — clone johwes/SCAR-test-c for the vuln sources
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLES_DIR="$SCRIPT_DIR/samples"
IR_DIR="$SCRIPT_DIR/ir"
FROM_REPO=0

for arg in "$@"; do
    [[ "$arg" == "--from-repo" ]] && FROM_REPO=1
done

if ! command -v clang &>/dev/null; then
    echo "error: clang not found — install clang or run inside the scar-agent container"
    exit 1
fi

mkdir -p "$IR_DIR"

if [[ "$FROM_REPO" == "1" ]]; then
    REPO_DIR="$(mktemp -d)"
    trap "rm -rf $REPO_DIR" EXIT
    echo "Cloning johwes/SCAR-test-c..."
    git clone --depth=1 https://github.com/johwes/SCAR-test-c "$REPO_DIR" -q
    VULN_DIR="$REPO_DIR"
else
    VULN_DIR="$SAMPLES_DIR"
fi

echo "Compiling vulnerable sources to LLVM IR..."
for src in "$VULN_DIR"/*.c; do
    base="$(basename "$src" .c)"
    # When using the repo, strip any _vuln suffix if present; files are plain bof.c etc.
    name="${base%_vuln}"
    ll="$IR_DIR/${name}_vuln.ll"
    clang -O0 -S -emit-llvm -Wno-everything -o "$ll" "$src" 2>/dev/null
    echo "  $(basename "$src") → $(basename "$ll")"
done

echo "Compiling fixed sources to LLVM IR..."
for src in "$SAMPLES_DIR"/*_fixed.c; do
    name="$(basename "$src" .c)"
    ll="$IR_DIR/$name.ll"
    clang -O0 -S -emit-llvm -Wno-everything -o "$ll" "$src" 2>/dev/null
    echo "  $(basename "$src") → $(basename "$ll")"
done

echo ""
python3 "$SCRIPT_DIR/demo.py" "$IR_DIR"
