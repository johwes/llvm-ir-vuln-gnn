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
    ("model.pt",              "model.pt",              "Block-level GNN classifier (55.52% Devign)"),
    ("model_instr.pt",        "model_instr.pt",         "Instruction-level GNN classifier (56.53% Devign)"),
    ("model_slice.pt",        "model_slice.pt",         "DFG slice GNN (55.60% Devign)"),
    ("model_slice_pdg.pt",    "model_slice_pdg.pt",     "PDG slice GNN (56.48% Devign)"),
    ("MODEL_CARD.md",         "README.md",              "Model card"),
    ("scan_ir.py",            "scan_ir.py",             "Inference CLI"),
    ("train.py",              "train.py",               "Training script (block-level)"),
    ("train_instr.py",        "train_instr.py",         "Training script (instruction-level)"),
    ("train_slice.py",        "train_slice.py",         "Training script (DFG slice)"),
    ("train_slice_pdg.py",    "train_slice_pdg.py",     "Training script (PDG slice)"),
    ("preprocess.py",         "preprocess.py",          "IR graph extractor (block-level)"),
    ("preprocess_instr.py",   "preprocess_instr.py",    "IR graph extractor (instruction-level)"),
    ("preprocess_slice.py",   "preprocess_slice.py",    "IR graph extractor (DFG slice)"),
    ("preprocess_slice_pdg.py","preprocess_slice_pdg.py","IR graph extractor (PDG slice)"),
    ("requirements.txt",      "requirements.txt",       "Python dependencies"),
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
