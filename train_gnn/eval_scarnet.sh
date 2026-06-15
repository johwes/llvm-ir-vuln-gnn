#!/usr/bin/env bash
# eval_scarnet.sh — Rank all scarnet functions by GNN vulnerability score.
#
# Compiles each scarnet source file to LLVM IR, scores every function with
# scan_ir.py --all-functions, and cross-references results against the known-
# vulnerable functions from the answer key.
#
# Usage:
#   bash eval_scarnet.sh
#   bash eval_scarnet.sh --model /path/to/model.pt
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$SCRIPT_DIR/model.pt"
SCANNER="$SCRIPT_DIR/scan_ir.py"
REPO_URL="https://github.com/johwes/scarnet.git"
WORKDIR="$(mktemp -d /tmp/scarnet-eval-XXXXXX)"
CLONE_DIR="$WORKDIR/scarnet"
IR_DIR="$WORKDIR/ir"
mkdir -p "$IR_DIR"

cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

for arg in "$@"; do
    case "$arg" in
        --model=*) MODEL="${arg#--model=}" ;;
        --model)   shift; MODEL="$1" ;;
    esac
done

PYTHON=$(command -v python || command -v python3)

# ── Known-vulnerable functions from answer key ───────────────────────────────
# Bugs 1-18 (planted + pipeline-discovered). Does NOT include clean helpers.
VULN_FNS="parse_cmd parse_batch handle_set handle_stats handle_del \
           scar_atoi scar_log scar_alloc_copy \
           session_login session_frag parse_msg_header session_consume_frag \
           handle_client"

is_vuln() {
    local fn="$1"
    for v in $VULN_FNS; do
        [[ "$fn" == "$v" ]] && return 0
    done
    return 1
}

# ── Prerequisites ─────────────────────────────────────────────────────────────
[[ -f "$MODEL"   ]] || { echo "ERROR: model.pt not found: $MODEL"; exit 1; }
command -v clang  &>/dev/null || { echo "ERROR: clang not in PATH"; exit 1; }

# ── Clone ─────────────────────────────────────────────────────────────────────
echo "Cloning johwes/scarnet ..."
git clone --quiet --depth 1 "$REPO_URL" "$CLONE_DIR"
echo ""

# ── Source files to compile ───────────────────────────────────────────────────
declare -a SOURCES=(
    "src/parse.c"
    "src/handler.c"
    "src/util.c"
    "src/session.c"
    "main.c"
)

# ── Compile and scan ──────────────────────────────────────────────────────────
# Collect: "score fn_name src_file"  lines for sorting
RESULTS_FILE="$WORKDIR/results.txt"
touch "$RESULTS_FILE"

for rel in "${SOURCES[@]}"; do
    c_file="$CLONE_DIR/$rel"
    base="${rel//\//_}"   # e.g. src_parse.c
    base="${base%.c}"
    ll_file="$IR_DIR/${base}.ll"

    printf "Compiling %-30s ... " "$rel"
    if ! clang -O0 -fno-inline -S -emit-llvm \
               -I "$CLONE_DIR/include" \
               -o "$ll_file" "$c_file" 2>/dev/null; then
        echo "FAILED (skip)"
        continue
    fi
    echo "ok"

    # Run scanner — each output line: "  fn_name   [N blocks]  XX.X%  ->  LABEL"
    raw=$("$PYTHON" "$SCANNER" "$ll_file" --model "$MODEL" --all-functions 2>/dev/null) || raw=""
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        fn=$(echo "$line"  | awk '{print $1}')
        score=$(echo "$line" | grep -oP '\d+\.\d+(?=%)' | head -1)
        [[ -z "$score" || -z "$fn" ]] && continue
        printf "%s %s %s\n" "$score" "$fn" "$rel" >> "$RESULTS_FILE"
    done <<< "$raw"
done

echo ""

# ── Print ranked table ────────────────────────────────────────────────────────
total_fns=0
vuln_in_top=0
total_vuln=$(echo $VULN_FNS | wc -w)

printf "%-5s %-40s %-20s %-8s %-12s %s\n" \
    "Rank" "Function" "File" "Score" "Prediction" "Known vuln?"
printf '%s\n' "$(printf '%0.s-' {1..95})"

rank=0
while IFS=' ' read -r score fn src; do
    rank=$((rank + 1))
    total_fns=$((total_fns + 1))

    prob_int=${score%.*}
    prob_dec=${score#*.}
    # prediction: score >= 50.0 -> VULNERABLE
    if awk "BEGIN{exit !($score >= 50.0)}"; then
        pred="VULNERABLE"
    else
        pred="safe"
    fi

    marker=""
    if is_vuln "$fn"; then
        marker="YES"
        if [[ $rank -le $total_vuln ]]; then
            vuln_in_top=$((vuln_in_top + 1))
        fi
    fi

    printf "%-5s %-40s %-20s %-8s %-12s %s\n" \
        "$rank" "$fn" "$src" "${score}%" "$pred" "$marker"
done < <(sort -rn "$RESULTS_FILE")

printf '%s\n' "$(printf '%0.s-' {1..95})"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Total functions scored: $total_fns"
echo "Known-vulnerable functions: $total_vuln"
echo "Known-vulnerable in top-${total_vuln}: ${vuln_in_top} / ${total_vuln}"
echo ""

pct=0
[[ $total_vuln -gt 0 ]] && pct=$((vuln_in_top * 100 / total_vuln))

if [[ $pct -ge 70 ]]; then
    echo "Strong signal: GNN ranks vulnerable functions near the top — pipeline integration warranted."
elif [[ $pct -ge 40 ]]; then
    echo "Moderate signal: GNN partially separates vulnerable from clean — useful as a weak ranker."
else
    echo "Weak signal: GNN scores do not correlate with known vulnerabilities on real code."
fi
