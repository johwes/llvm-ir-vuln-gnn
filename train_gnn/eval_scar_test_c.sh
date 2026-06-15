#!/usr/bin/env bash
# eval_scar_test_c.sh — Smoke-test the block-level GNN against SCAR-test-c.
#
# Compiles all 7 known-vulnerable C files in johwes/scar-test-c to LLVM IR,
# runs scan_ir.py on each, and checks whether the model predicts VULNERABLE.
# Adds one trivially clean synthetic function as a negative control.
#
# Usage:
#   bash eval_scar_test_c.sh
#   bash eval_scar_test_c.sh --model /path/to/other.pt
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-$SCRIPT_DIR/model.pt}"

# --model flag support
for arg in "$@"; do
    case "$arg" in
        --model) shift; MODEL="$1"; shift ;;
        --model=*) MODEL="${arg#--model=}" ;;
    esac
done

SCANNER="$SCRIPT_DIR/scan_ir.py"
REPO_URL="https://github.com/johwes/scar-test-c.git"
WORKDIR="$(mktemp -d /tmp/scar-eval-XXXXXX)"
CLONE_DIR="$WORKDIR/scar-test-c"
IR_DIR="$WORKDIR/ir"
mkdir -p "$IR_DIR"

cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

# ── Prerequisites ────────────────────────────────────────────────────────────

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model not found: $MODEL" >&2
    echo "  Run train.py first to generate model.pt" >&2
    exit 1
fi
if ! command -v clang &>/dev/null; then
    echo "ERROR: clang not found in PATH" >&2
    exit 1
fi
if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
    echo "ERROR: python not found in PATH" >&2
    exit 1
fi
PYTHON=$(command -v python || command -v python3)

# ── Clone ────────────────────────────────────────────────────────────────────

echo "Cloning johwes/scar-test-c ..."
git clone --quiet --depth 1 "$REPO_URL" "$CLONE_DIR" 2>&1 | grep -v "^$" || true
echo ""

# ── Ground-truth table ───────────────────────────────────────────────────────
# Format: file:CWE:expected_label
declare -a GROUND_TRUTH=(
    "doublefree.c:CWE-415:VULNERABLE"
    "nullderef.c:CWE-476:VULNERABLE"
    "oob_read.c:CWE-125:VULNERABLE"
    "uninit.c:CWE-457:VULNERABLE"
    "divzero.c:CWE-369:VULNERABLE"
    "bof.c:CWE-121:VULNERABLE"
    "signedoverflow.c:CWE-190:VULNERABLE"
    "clean.c:none:safe"
)

# ── Synthetic clean function ─────────────────────────────────────────────────
cat > "$WORKDIR/clean.c" << 'CEOF'
int add(int a, int b) { return a + b; }
CEOF

# ── Scan helper ──────────────────────────────────────────────────────────────

correct=0
total=0

scan_one() {
    local c_file="$1"
    local base
    base="$(basename "$c_file")"
    local name="${base%.c}"
    local cwe="$2"
    local expected="$3"
    local ll_file="$IR_DIR/${name}.ll"

    # Compile
    if ! clang -O1 -S -emit-llvm -o "$ll_file" "$c_file" 2>/dev/null; then
        printf "%-20s %-10s %-10s %-8s %-12s %s\n" \
            "$base" "$cwe" "$expected" "ERR" "compile-fail" "?"
        total=$((total + 1))
        return
    fi

    # Scan — last word is label, 4th field is score
    local raw
    raw=$("$PYTHON" "$SCANNER" "$ll_file" --model "$MODEL" 2>/dev/null) || raw=""

    local score pred
    score=$(echo "$raw" | awk '{print $4}')   # e.g. "72.3%"
    pred=$(echo "$raw"  | awk '{print $NF}')  # VULNERABLE or safe

    local ok="NO"
    if [[ "$pred" == "$expected" ]]; then
        ok="YES"
        correct=$((correct + 1))
    fi
    total=$((total + 1))

    printf "%-20s %-10s %-10s %-8s %-12s %s\n" \
        "$base" "$cwe" "$expected" "${score:---%}" "${pred:---}" "$ok"
}

# ── Run ──────────────────────────────────────────────────────────────────────

printf "%-20s %-10s %-10s %-8s %-12s %s\n" \
    "File" "CWE" "Expected" "Score" "Prediction" "Correct?"
printf '%s\n' "------------------------------------------------------------------------"

for entry in "${GROUND_TRUTH[@]}"; do
    IFS=':' read -r fname cwe expected <<< "$entry"
    if [[ "$fname" == "clean.c" ]]; then
        scan_one "$WORKDIR/clean.c" "$cwe" "$expected"
    elif [[ -f "$CLONE_DIR/$fname" ]]; then
        scan_one "$CLONE_DIR/$fname" "$cwe" "$expected"
    else
        printf "%-20s %-10s %-10s %-8s %-12s %s\n" \
            "$fname" "$cwe" "$expected" "---" "not found" "?"
        total=$((total + 1))
    fi
done

printf '%s\n' "------------------------------------------------------------------------"
printf 'Result: %d / %d correct\n\n' "$correct" "$total"

if [[ $correct -eq $total ]]; then
    echo "All correct — model generalises to real compiled IR."
elif [[ $correct -ge 6 ]]; then
    echo "Strong generalisation — minor misses expected across CWE types."
elif [[ $correct -ge 4 ]]; then
    echo "Partial generalisation — model captures some CWE patterns."
else
    echo "Weak generalisation — model may have memorised Devign-specific patterns."
fi
