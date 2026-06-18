#!/usr/bin/env python3
"""
hf_upload.py — Upload GNN defect detector checkpoints to HuggingFace Hub.

Usage:
    pip install huggingface_hub
    huggingface-cli login          # paste your HF token when prompted
    python hf_upload.py            # uploads model.pt + model_instr.pt + MODEL_CARD.md
    python hf_upload.py --dry-run  # list files that would be uploaded
"""
import argparse
import sys
from pathlib import Path

HERE    = Path(__file__).parent
REPO_ID = "johnnywesterlund/scar-gnn-defect-detector"

UPLOADS = [
    # Model checkpoints
    ("model.pt",                  "model.pt",                  "Block-level GNN §4d (55.52% Devign, 9/13 scarnet)"),
    ("model_instr.pt",            "model_instr.pt",            "Instruction-level GNN §7 (56.53% Devign, 9/13 scarnet)"),
    ("model_instr_v2.pt",         "model_instr_v2.pt",         "Perfograph + call categories §13 (58.75% Devign, 10/13 scarnet)"),
    ("model_instr_v3.pt",         "model_instr_v3.pt",         "VSDG memory ordering edges §14 (57.47% Devign, 10/13 scarnet)"),
    ("model_instr_v4.pt",         "model_instr_v4.pt",         "Register name embedding §15 (57.47% Devign, 8/13 scarnet)"),
    ("model_instr_v5.pt",         "model_instr_v5.pt",         "Static analysis flags §16 (57.15% Devign, 8/13 scarnet)"),
    ("model_instr_v6.pt",         "model_instr_v6.pt",         "Taint propagation §17 (58.00% Devign, 10/13 scarnet)"),
    ("model_slice.pt",            "model_slice.pt",            "DFG slice GNN §11 (55.60% Devign, 10/13 scarnet)"),
    ("model_slice_pdg.pt",        "model_slice_pdg.pt",        "PDG slice GNN §12 — recommended (56.48% Devign, 11/13 scarnet)"),
    ("model_slice_pdg_v2.pt",     "model_slice_pdg_v2.pt",     "PDG + taint flags §22 (9/13 scarnet)"),
    ("model_slice_pdg_v3.pt",     "model_slice_pdg_v3.pt",     "PDG sink-node readout §23 (55.40% Devign, 9/13 scarnet)"),
    ("model_slice_pdg_v4.pt",     "model_slice_pdg_v4.pt",     "PDG + intrinsic-aware sinks §24 (TBD Devign, TBD scarnet)"),
    # Model card
    ("MODEL_CARD.md",             "README.md",                 "Model card"),
    # Inference + context enrichment
    ("scan_ir.py",                "scan_ir.py",                "Inference CLI (--context flag for LLM prompt injection)"),
    ("slice_context.py",          "slice_context.py",          "PDG slice → LLM vulnerability context"),
    # Training scripts
    ("train.py",                  "train.py",                  "Training script §4d (block-level)"),
    ("train_instr.py",            "train_instr.py",            "Training script §7 (instruction-level baseline)"),
    ("train_instr_v2.py",         "train_instr_v2.py",         "Training script §13 (Perfograph + call targets)"),
    ("train_instr_v3.py",         "train_instr_v3.py",         "Training script §14 (VSDG memory ordering edges)"),
    ("train_instr_v4.py",         "train_instr_v4.py",         "Training script §15 (register name embedding)"),
    ("train_instr_v5.py",         "train_instr_v5.py",         "Training script §16 (static analysis flags)"),
    ("train_instr_v6.py",         "train_instr_v6.py",         "Training script §17 (taint propagation)"),
    ("train_slice.py",            "train_slice.py",            "Training script §11 (DFG slice)"),
    ("train_slice_pdg.py",        "train_slice_pdg.py",        "Training script §12 (PDG slice — recommended)"),
    ("train_slice_pdg_v2.py",     "train_slice_pdg_v2.py",     "Training script §22 (PDG + taint flags)"),
    ("train_slice_pdg_v3.py",     "train_slice_pdg_v3.py",     "Training script §23 (sink-node readout + residual/LN)"),
    ("train_slice_pdg_v4.py",     "train_slice_pdg_v4.py",     "Training script §24 (intrinsic-aware sink retraining)"),
    # Preprocessing scripts
    ("preprocess.py",             "preprocess.py",             "IR graph extractor (block-level)"),
    ("preprocess_instr.py",       "preprocess_instr.py",       "IR graph extractor §7 (instruction-level baseline)"),
    ("preprocess_instr_v2.py",    "preprocess_instr_v2.py",    "IR graph extractor §13 (Perfograph + call targets)"),
    ("preprocess_instr_v3.py",    "preprocess_instr_v3.py",    "IR graph extractor §14 (VSDG state edges)"),
    ("preprocess_instr_v4.py",    "preprocess_instr_v4.py",    "IR graph extractor §15 (register name embedding)"),
    ("preprocess_instr_v5.py",    "preprocess_instr_v5.py",    "IR graph extractor §16 (static analysis flags)"),
    ("preprocess_instr_v6.py",    "preprocess_instr_v6.py",    "IR graph extractor §17 (taint propagation)"),
    ("preprocess_slice.py",       "preprocess_slice.py",       "IR graph extractor §11 (DFG slice)"),
    ("preprocess_slice_pdg.py",   "preprocess_slice_pdg.py",   "IR graph extractor §12 (PDG slice)"),
    ("preprocess_slice_pdg_v3.py","preprocess_slice_pdg_v3.py","IR graph extractor §23 (PDG v3, sink_mask + CD cap)"),
    ("requirements.txt",          "requirements.txt",          "Python dependencies"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repo", default=REPO_ID)
    args = ap.parse_args()

    missing = [src for src, _, _ in UPLOADS if not (HERE / src).exists()]
    if missing:
        print(f"WARNING: missing files (will skip): {missing}")

    if args.dry_run:
        print(f"Would upload to: https://huggingface.co/{args.repo}")
        for src, dst, desc in UPLOADS:
            exists = (HERE / src).exists()
            size   = f"{(HERE / src).stat().st_size / 1024:.0f} KB" if exists else "MISSING"
            print(f"  {'OK' if exists else '--'}  {src:30s} -> {dst:30s}  {size}  ({desc})")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    print(f"Uploading to: https://huggingface.co/{args.repo}\n")

    for src, dst, desc in UPLOADS:
        local = HERE / src
        if not local.exists():
            print(f"  SKIP  {src} (not found)")
            continue
        size = local.stat().st_size / 1024
        print(f"  -> {dst}  ({size:.0f} KB)  {desc}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=dst,
            repo_id=args.repo,
        )

    print(f"\nDone. View at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
