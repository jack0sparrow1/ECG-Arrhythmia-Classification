from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .train import main as train_main


def _read_macro_f1(run_dir: Path) -> float:
    report = json.loads((run_dir / "classification_report.json").read_text())
    return float(report["macro avg"]["f1-score"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="artifacts/compare")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--use-class-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run MLP
    mlp_dir = out_dir / "mlp"
    _run_args = [
        "src.train",
        "--csv",
        args.csv,
        "--artifact-dir",
        str(mlp_dir),
        "--arch",
        "mlp",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
    ]
    if args.use_class_weights:
        _run_args.append("--use-class-weights")

    import sys

    old_argv = sys.argv
    try:
        sys.argv = _run_args
        train_main()
    finally:
        sys.argv = old_argv

    # Run CNN+BiLSTM
    cnn_dir = out_dir / "cnn_bilstm"
    _run_args = [
        "src.train",
        "--csv",
        args.csv,
        "--artifact-dir",
        str(cnn_dir),
        "--arch",
        "cnn_bilstm",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
    ]
    if args.use_class_weights:
        _run_args.append("--use-class-weights")

    old_argv = sys.argv
    try:
        sys.argv = _run_args
        train_main()
    finally:
        sys.argv = old_argv

    mlp_f1 = _read_macro_f1(mlp_dir)
    cnn_f1 = _read_macro_f1(cnn_dir)
    print("Macro F1 comparison")
    print(" - MLP:", mlp_f1)
    print(" - CNN+BiLSTM:", cnn_f1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

