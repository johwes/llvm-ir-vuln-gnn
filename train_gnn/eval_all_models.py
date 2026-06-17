#!/usr/bin/env python3
"""
eval_all_models.py — Score every trained checkpoint against a set of IR functions.

Runs all registered model checkpoints on each function in the target IR, ranks
by score, and (optionally) computes top-K precision/recall against an answer key.

Usage:
    # Auto-clone johwes/scarnet, compile to IR, and score (requires git + clang)
    python eval_all_models.py --scarnet --answer-key answer-key.txt

    # Same, but keep the compiled IR for reuse on the next run
    python eval_all_models.py --scarnet --keep-ir /tmp/scarnet-ir/ --answer-key answer-key.txt

    # Reuse previously compiled IR (skip clone+compile)
    python eval_all_models.py --ir-dir /tmp/scarnet-ir/ --answer-key answer-key.txt

    # Show only the summary table, not per-model ranked lists
    python eval_all_models.py --scarnet --answer-key answer-key.txt --summary-only

    # Download the 4 published HuggingFace checkpoints to --model-dir, then score
    python eval_all_models.py --scarnet --fetch-hf --answer-key answer-key.txt

Answer key format: plain text, one function name per line (exact match against IR @name).
Lines starting with # are treated as comments and ignored.

Each model in the registry requires a different graph format, so the script runs
the correct preprocessor for each checkpoint automatically.

HuggingFace models (--fetch-hf downloads these automatically):
    model.pt          §4d  block-level DefectGNN
    model_instr.pt    §7   instruction-level baseline
    model_slice.pt    §11  DFG slice GNN
    model_slice_pdg.pt §12 PDG slice GNN

Locally retrained models (not on HuggingFace — run the matching train script):
    model_instr_v2.pt   python train_instr_v2.py   (needs data/*_instr_v2_graphs.pkl)
    model_instr_v3.pt   python train_instr_v3.py   (needs data/*_instr_v3_graphs.pkl)
    model_instr_v4.pt   python train_instr_v4.py   (needs data/*_instr_v4_graphs.pkl)
    model_instr_v5.pt   python train_instr_v5.py   (needs data/*_instr_v5_graphs.pkl)
    model_instr_v6.pt   python train_instr_v6.py   (needs data/*_instr_v6_graphs.pkl)
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# ---------------------------------------------------------------------------
# Import preprocessors under unambiguous aliases
# ---------------------------------------------------------------------------

from preprocess import ir_to_graph as _pp_block

from preprocess_instr import ir_to_graph_instr as _pp_instr_v1
from preprocess_instr_v2 import ir_to_graph_instr as _pp_instr_v2
from preprocess_instr_v3 import ir_to_graph_instr as _pp_instr_v3
from preprocess_instr_v4 import ir_to_graph_instr as _pp_instr_v4
from preprocess_instr_v5 import ir_to_graph_instr as _pp_instr_v5
from preprocess_instr_v6 import ir_to_graph_instr as _pp_instr_v6

from preprocess_slice     import ir_to_graph_slice     as _pp_slice
from preprocess_slice_pdg import ir_to_graph_slice_pdg as _pp_slice_pdg

# ---------------------------------------------------------------------------
# Import model modules under unambiguous aliases
# ---------------------------------------------------------------------------

import train            as _m_block
import train_instr      as _m_v1
import train_instr_v2   as _m_v2
import train_instr_v3   as _m_v3
import train_instr_v4   as _m_v4
import train_instr_v5   as _m_v5
import train_instr_v6   as _m_v6
import train_slice      as _m_slice
import train_slice_pdg  as _m_pdg


# ---------------------------------------------------------------------------
# Model loaders — each infers hyper-parameters from the checkpoint tensor shapes
# ---------------------------------------------------------------------------

def _load_block(path: Path) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    hidden = ckpt["lin.weight"].shape[1]
    m = _m_block.DefectGNN(_m_block.N_FEATURES, hidden=hidden)
    m.load_state_dict(ckpt)
    return m.eval()


def _load_instr_auto(path: Path) -> tuple[nn.Module, object]:
    """Load any InstructionGNN checkpoint by detecting the architecture from weights.

    Returns (model, preprocess_fn) — the preprocessor that matches the detected
    architecture, which may differ from the registry entry if the checkpoint was
    saved under the wrong filename.

    Detected variants:
      has name_embed.weight          → v4 (register name embedding)
      num_relations == 4             → v3 (VSDG state edges)
      conv1_in == embed_dim          → v1 (opcode only)
      conv1_in == embed_dim + 1      → v2 (+ constant magnitude)
      conv1_in == embed_dim + 2      → v5/v6 (+ static flag / taint)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    vocab_size    = ckpt["embed.weight"].shape[0]
    embed_dim     = ckpt["embed.weight"].shape[1]
    # RGCNConv weight shape: (num_relations, in_channels, out_channels)
    num_relations = ckpt["conv1.weight"].shape[0]
    conv1_in      = ckpt["conv1.weight"].shape[1]   # in_channels (NOT shape[2])
    # lin = Linear(hidden, 1) → weight shape (1, hidden); safest way to read hidden
    hidden        = ckpt["lin.weight"].shape[1]
    extra         = conv1_in - embed_dim
    has_name      = "name_embed.weight" in ckpt

    if has_name:
        name_embed_dim = ckpt["name_embed.weight"].shape[1]
        variant    = f"v4 (name_embed, in={conv1_in}, rels={num_relations})"
        m          = _m_v4.InstructionGNN(vocab_size, embed_dim, hidden, name_embed_dim)
        preprocess = _pp_instr_v4
    elif num_relations == 4:
        variant    = f"v3 (VSDG state edges, in={conv1_in}, rels=4)"
        m          = _m_v3.InstructionGNN(vocab_size, embed_dim, hidden)
        preprocess = _pp_instr_v3
    elif extra == 0:
        variant    = f"v1 (opcode only, in={conv1_in}, rels={num_relations})"
        m          = _m_v1.InstructionGNN(vocab_size, embed_dim, hidden)
        preprocess = _pp_instr_v1
    elif extra == 1:
        variant    = f"v2/§13 (+ const_mag, in={conv1_in}, rels={num_relations})"
        m          = _m_v2.InstructionGNN(vocab_size, embed_dim, hidden)
        preprocess = _pp_instr_v2
    elif extra == 2:
        variant    = f"v5/v6 (+ flag/taint, in={conv1_in}, rels={num_relations})"
        m          = _m_v5.InstructionGNN(vocab_size, embed_dim, hidden)
        preprocess = _pp_instr_v6   # v6 is a superset; works for both v5 and v6 weights
    else:
        raise ValueError(
            f"Unrecognised InstructionGNN shape: vocab={vocab_size}, "
            f"embed={embed_dim}, hidden={hidden}, conv1_in={conv1_in}, "
            f"relations={num_relations}")

    print(f"    detected {variant}, vocab={vocab_size}, hidden={hidden}")
    m.load_state_dict(ckpt)
    return m.eval(), preprocess


def _load_slice(path: Path) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    vocab_size = ckpt["embed.weight"].shape[0]
    embed_dim  = ckpt["embed.weight"].shape[1]
    hidden     = ckpt["lin.weight"].shape[1]   # Linear(hidden,1) → weight (1,hidden)
    m = _m_slice.SliceGNN(vocab_size, embed_dim, hidden)
    m.load_state_dict(ckpt)
    return m.eval()


def _load_pdg(path: Path) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    vocab_size = ckpt["embed.weight"].shape[0]
    embed_dim  = ckpt["embed.weight"].shape[1]
    hidden     = ckpt["lin.weight"].shape[1]
    m = _m_pdg.SlicePDGGNN(vocab_size, embed_dim, hidden)
    m.load_state_dict(ckpt)
    return m.eval()


# ---------------------------------------------------------------------------
# Model registry
# Each entry: checkpoint filename, human label, Devign accuracy, preprocessor,
# model loader.  Add new entries here as new checkpoints are trained.
#
# Note: model_gru.pt is excluded — it stores opcode sequences inside the graph
# data structure (not just opcode IDs), requiring a different loading path that
# is incompatible with the standard preprocess → score pipeline used here.
# ---------------------------------------------------------------------------

REGISTRY = [
    {
        "checkpoint": "model.pt",
        "label":      "§4d  block-level DefectGNN",
        "devign":     "55.52%",
        "preprocess": _pp_block,
        "load_model": _load_block,
    },
    {
        "checkpoint": "model_instr.pt",
        "label":      "§7   instr baseline (opcode only)",
        "devign":     "56.53%",
        "preprocess": _pp_instr_v1,
        "load_model": _load_instr_auto,
    },
    {
        "checkpoint": "model_instr_v2.pt",
        "label":      "§13  Perfograph + call categories",
        "devign":     "58.75%",
        "preprocess": _pp_instr_v2,
        "load_model": _load_instr_auto,
    },
    {
        "checkpoint": "model_instr_v3.pt",
        "label":      "§14  VSDG memory ordering edges",
        "devign":     "57.47%",
        "preprocess": _pp_instr_v3,
        "load_model": _load_instr_auto,
    },
    {
        "checkpoint": "model_instr_v4.pt",
        "label":      "§15  register name embedding",
        "devign":     "57.47%",
        "preprocess": _pp_instr_v4,
        "load_model": _load_instr_auto,
    },
    {
        "checkpoint": "model_instr_v5.pt",
        "label":      "§16  static analysis flags",
        "devign":     "57.15%",
        "preprocess": _pp_instr_v5,
        "load_model": _load_instr_auto,
    },
    {
        "checkpoint": "model_instr_v6.pt",
        "label":      "§17  taint propagation",
        "devign":     "pending",
        "preprocess": _pp_instr_v6,
        "load_model": _load_instr_auto,
    },
    {
        "checkpoint": "model_slice.pt",
        "label":      "§11  DFG slice GNN",
        "devign":     "55.60%",
        "preprocess": _pp_slice,
        "load_model": _load_slice,
    },
    {
        "checkpoint": "model_slice_pdg.pt",
        "label":      "§12  PDG slice GNN",
        "devign":     "56.48%",
        "preprocess": _pp_slice_pdg,
        "load_model": _load_pdg,
    },
]


# ---------------------------------------------------------------------------
# IR parsing
# ---------------------------------------------------------------------------

_FN_NAME_RE = re.compile(r'@([\w.$]+)\s*\(')


def _split_functions(ir_text: str) -> list[tuple[str, str]]:
    """Split a .ll file into (fn_name, fn_ir_text) pairs, one per define."""
    segs = re.split(r'(?=^define\b)', ir_text, flags=re.MULTILINE)
    result = []
    for seg in segs:
        seg = seg.strip()
        if not seg.startswith("define"):
            continue
        m = _FN_NAME_RE.search(seg[:300])
        if not m:
            continue
        result.append((m.group(1), seg))
    return result


def _collect_functions(path: Path) -> list[tuple[str, str]]:
    """Collect (fn_name, fn_ir) from a directory of .ll files or a single .ll file."""
    if path.is_dir():
        fns = []
        for ll in sorted(path.glob("*.ll")):
            fns.extend(_split_functions(ll.read_text(errors="replace")))
        return fns
    return _split_functions(path.read_text(errors="replace"))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(model: nn.Module, g: dict) -> float:
    x  = torch.tensor(g["x"])  # preserve dtype: Long for v1, float for v2+
    if x.is_floating_point():
        x = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
    ei = torch.tensor(g["edge_index"], dtype=torch.long)
    et = torch.tensor(g["edge_type"],  dtype=torch.long)
    batch = torch.zeros(x.shape[0], dtype=torch.long)
    with torch.no_grad():
        return torch.sigmoid(model(x, ei, et, batch)).item()


def _run_model(entry: dict, functions: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Score all functions with one model. Returns [(fn_name, score), ...] sorted desc."""
    results = []
    preprocess = entry["_preprocess"]   # set by main() after load; accounts for auto-detected arch
    model      = entry["_model"]
    for fn_name, fn_ir in functions:
        g = preprocess(fn_ir)
        if g is None:
            continue
        score = _score(model, g)
        results.append((fn_name, score))
    results.sort(key=lambda r: r[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Answer key
# ---------------------------------------------------------------------------

def _load_answer_key(path: Path) -> set[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return {ln for ln in lines if ln and not ln.startswith("#")}


def _precision_recall(ranked: list[tuple[str, float]],
                      answer_key: set[str],
                      k: int) -> tuple[int, float, float]:
    top_k = {fn for fn, _ in ranked[:k]}
    hits  = top_k & answer_key
    n     = len(hits)
    prec  = n / k if k else 0.0
    rec   = n / len(answer_key) if answer_key else 0.0
    return n, prec, rec


# ---------------------------------------------------------------------------
# HuggingFace model download
# ---------------------------------------------------------------------------

_HF_REPO = "johnnywesterlund/scar-gnn-defect-detector"

# The 4 published checkpoints. v2–v6 were trained after the last HF upload
# and must be retrained locally using their respective train_instr_vN.py scripts.
_HF_CHECKPOINTS = [
    "model.pt",
    "model_instr.pt",
    "model_slice.pt",
    "model_slice_pdg.pt",
]


def _fetch_hf_models(dest_dir: Path) -> None:
    """Download published checkpoints from HuggingFace Hub into dest_dir."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    print(f"Fetching {len(_HF_CHECKPOINTS)} checkpoints from {_HF_REPO} ...")
    for filename in _HF_CHECKPOINTS:
        dest = dest_dir / filename
        if dest.exists():
            print(f"  already exists, skipping: {filename}")
            continue
        print(f"  downloading {filename} ...", end=" ", flush=True)
        try:
            hf_hub_download(
                repo_id=_HF_REPO,
                filename=filename,
                local_dir=str(dest_dir),
            )
            print("ok")
        except Exception as e:
            print(f"FAILED: {e}")
    print()


# ---------------------------------------------------------------------------
# Scarnet clone + compile
# ---------------------------------------------------------------------------

_SCARNET_REPO  = "https://github.com/johwes/scarnet.git"
_SCARNET_SRCS  = [
    "src/parse.c",
    "src/handler.c",
    "src/util.c",
    "src/session.c",
    "main.c",
]


def _setup_scarnet_ir(keep_ir: Path | None) -> tuple[Path, Path]:
    """Clone johwes/scarnet and compile every C source to LLVM IR.

    Returns (ir_dir, tmpdir) where tmpdir must be deleted by the caller when done.
    If keep_ir is given, IR files are written there (persistent); otherwise they go
    inside tmpdir alongside the clone and are removed with it.
    """
    tmpdir    = Path(tempfile.mkdtemp(prefix="scarnet-eval-"))
    clone_dir = tmpdir / "scarnet"
    ir_dir    = keep_ir if keep_ir else tmpdir / "ir"
    ir_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cloning {_SCARNET_REPO} ...")
    try:
        subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1",
             _SCARNET_REPO, str(clone_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"git clone failed: {e}") from e

    compiled = 0
    for rel in _SCARNET_SRCS:
        c_file = clone_dir / rel
        base   = rel.replace("/", "_").removesuffix(".c")
        ll_out = ir_dir / f"{base}.ll"

        print(f"  compiling {rel:<30} ...", end=" ", flush=True)
        result = subprocess.run(
            ["clang", "-O0", "-fno-inline", "-S", "-emit-llvm",
             "-I", str(clone_dir / "include"),
             "-o", str(ll_out), str(c_file)],
            capture_output=True,
        )
        if result.returncode == 0:
            print("ok")
            compiled += 1
        else:
            print("FAILED (skipped)")
            if result.stderr:
                print(f"    {result.stderr.decode(errors='replace').strip()}")

    print(f"  {compiled}/{len(_SCARNET_SRCS)} source files compiled → {ir_dir}\n")
    return ir_dir, tmpdir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scarnet", action="store_true",
                     help="Clone johwes/scarnet and compile to IR automatically "
                          "(requires git and clang in PATH)")
    src.add_argument("--ir-dir",
                     help=".ll file or directory of .ll files to evaluate directly")

    ap.add_argument("--fetch-hf",    action="store_true",
                    help="Download the 4 published checkpoints from HuggingFace before "
                         "scoring (requires: pip install huggingface_hub)")
    ap.add_argument("--keep-ir",     default=None, metavar="DIR",
                    help="With --scarnet: save compiled IR to this directory so you "
                         "can reuse it with --ir-dir on later runs")
    ap.add_argument("--answer-key",  default=None,
                    help="Text file with one known-vulnerable function name per line "
                         "(lines starting with # are ignored)")
    ap.add_argument("--top-k",       type=int, default=None,
                    help="K for top-K P/R (default: number of lines in answer key)")
    ap.add_argument("--model-dir",   default=str(HERE),
                    help="Directory containing .pt checkpoint files (default: script dir)")
    ap.add_argument("--summary-only", action="store_true",
                    help="Print only the summary table, skip per-model ranked lists")
    args = ap.parse_args()

    if args.keep_ir and not args.scarnet:
        ap.error("--keep-ir only makes sense with --scarnet")

    model_dir = Path(args.model_dir)
    tmpdir: Path | None = None

    # -- Fetch HuggingFace models -------------------------------------------------
    if args.fetch_hf:
        model_dir.mkdir(parents=True, exist_ok=True)
        _fetch_hf_models(model_dir)

    # -- Resolve IR source -------------------------------------------------------
    if args.scarnet:
        for tool in ("git", "clang"):
            if not shutil.which(tool):
                print(f"ERROR: {tool} not found in PATH (required for --scarnet)")
                sys.exit(1)
        keep_ir = Path(args.keep_ir) if args.keep_ir else None
        ir_path, tmpdir = _setup_scarnet_ir(keep_ir)
    else:
        ir_path = Path(args.ir_dir)
        if not ir_path.exists():
            print(f"ERROR: --ir-dir {ir_path} does not exist")
            sys.exit(1)

    # Load answer key
    answer_key: set[str] = set()
    top_k = args.top_k
    if args.answer_key:
        answer_key = _load_answer_key(Path(args.answer_key))
        if top_k is None:
            top_k = len(answer_key)
        print(f"Answer key: {len(answer_key)} known-vulnerable functions  (top-K = {top_k})")

    # Collect IR functions
    functions = _collect_functions(ir_path)
    if not functions:
        print(f"ERROR: no parseable functions found in {ir_path}")
        sys.exit(1)
    print(f"Functions: {len(functions)} found in {ir_path}")
    print()

    # Filter registry to checkpoints that exist on disk
    active = []
    for entry in REGISTRY:
        ckpt_path = model_dir / entry["checkpoint"]
        if not ckpt_path.exists():
            print(f"  SKIP  {entry['checkpoint']} (not found in {model_dir})")
            continue
        active.append({**entry, "_ckpt_path": ckpt_path})
    if not active:
        print("ERROR: no checkpoints found. Check --model-dir.")
        sys.exit(1)
    print(f"Running {len(active)} model(s)...\n")

    # Load models and score
    summary_rows = []
    for entry in active:
        label = entry["label"]
        print(f"[{entry['checkpoint']}]  {label}")
        try:
            result = entry["load_model"](entry["_ckpt_path"])
            # _load_instr_auto returns (model, preprocess_fn); other loaders return model only
            if isinstance(result, tuple):
                entry["_model"], detected_pp = result
                if detected_pp is not entry["preprocess"]:
                    print(f"  NOTE: using detected preprocessor (checkpoint architecture "
                          f"differs from registry — file was likely overwritten)")
                entry["_preprocess"] = detected_pp
            else:
                entry["_model"]      = result
                entry["_preprocess"] = entry["preprocess"]
        except Exception as e:
            print(f"  ERROR loading checkpoint: {e}\n")
            continue

        ranked = _run_model(entry, functions)
        entry["_ranked"] = ranked
        scored = len(ranked)
        print(f"  scored {scored} / {len(functions)} functions")

        row = {
            "checkpoint": entry["checkpoint"],
            "label":      label,
            "devign":     entry["devign"],
            "scored":     scored,
            "hits":       None,
            "prec":       None,
            "rec":        None,
            "ranked":     ranked,
        }

        if answer_key and top_k and ranked:
            n, prec, rec = _precision_recall(ranked, answer_key, top_k)
            row["hits"] = n
            row["prec"] = prec
            row["rec"]  = rec
            print(f"  top-{top_k}: {n}/{len(answer_key)} hits  "
                  f"P={prec:.1%}  R={rec:.1%}")
        print()
        summary_rows.append(row)

    # Per-model ranked lists
    if not args.summary_only:
        for row in summary_rows:
            ranked = row["ranked"]
            if not ranked:
                continue
            print(f"=== {row['checkpoint']}  ({row['label']}) ===")
            boundary = top_k if top_k else len(ranked)
            print(f"  {'Rank':>4}  {'Function':<45}  {'Score':>6}  {'Vuln?':>5}")
            print(f"  {'----':>4}  {'-'*45}  {'------':>6}  {'-----':>5}")
            for i, (fn, score) in enumerate(ranked, 1):
                if i == boundary + 1:
                    print(f"  {'----':>4}  {'-'*45}  {'------':>6}  (below top-{boundary})")
                vuln_marker = ""
                if answer_key:
                    vuln_marker = "YES" if fn in answer_key else "no"
                print(f"  {i:>4}  {fn:<45}  {score:>5.1%}  {vuln_marker:>5}")
            print()

    # Summary table
    if not summary_rows:
        return

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    hdr_pr = f"  {'Hits':>6}  {'P@K':>6}  {'R@K':>6}" if answer_key else ""
    print(f"  {'Checkpoint':<24}  {'Section':<35}  {'Devign':>7}{hdr_pr}")
    print(f"  {'-'*24}  {'-'*35}  {'-'*7}" + ("  " + "-"*6 + "  " + "-"*6 + "  " + "-"*6 if answer_key else ""))
    for row in summary_rows:
        pr_cols = ""
        if answer_key:
            hits_str = f"{row['hits']}/{len(answer_key)}" if row['hits'] is not None else "—"
            prec_str = f"{row['prec']:.1%}" if row['prec'] is not None else "—"
            rec_str  = f"{row['rec']:.1%}"  if row['rec']  is not None else "—"
            pr_cols = f"  {hits_str:>6}  {prec_str:>6}  {rec_str:>6}"
        print(f"  {row['checkpoint']:<24}  {row['label']:<35}  {row['devign']:>7}{pr_cols}")
    if answer_key:
        print(f"\n  K = {top_k}")

    # -- Cleanup temp clone+IR dir -----------------------------------------------
    if tmpdir and tmpdir.exists():
        if args.keep_ir:
            # Only the clone is in tmpdir; IR was written to keep_ir
            print(f"(cleaned up temporary clone dir)")
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
