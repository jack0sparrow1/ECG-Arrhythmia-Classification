from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import apply_smote, compute_class_weights, load_csv_as_sequence
from .model import build_cnn_bilstm, build_mlp


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to dataset CSV")
    ap.add_argument("--artifact-dir", type=str, default="artifacts/run", help="Output directory")
    ap.add_argument("--arch", type=str, default="cnn_bilstm", choices=["mlp", "cnn_bilstm"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mlp-hidden", type=int, default=128, help="MLP hidden size (when --arch mlp)")
    ap.add_argument("--use-class-weights", action="store_true", help="Use weighted CE (recommended)")
    ap.add_argument("--use-smote", action="store_true", help="Apply SMOTE on train set (optional)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = _ensure_dir(Path(args.artifact_dir))

    splits = load_csv_as_sequence(args.csv, random_state=args.seed)
    x_train, y_train = splits.x_train, splits.y_train
    x_val, y_val = splits.x_val, splits.y_val

    if args.use_smote:
        x_train, y_train = apply_smote(x_train, y_train, random_state=args.seed)

    num_classes = int(np.max(y_train)) + 1
    seq_len = int(x_train.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.arch == "mlp":
        model = build_mlp(seq_len=seq_len, num_classes=num_classes, hidden=args.mlp_hidden).to(device)
    else:
        model = build_cnn_bilstm(seq_len=seq_len, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.use_class_weights:
        cw = compute_class_weights(y_train)
        weight = torch.tensor([cw.get(i, 1.0) for i in range(num_classes)], dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "val_f1_per_class": []}
    best_val_macro_f1 = -1.0
    patience = 5
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} [train]", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses: list[float] = []
        correct = 0
        total = 0
        val_pred_parts: list[np.ndarray] = []
        val_true_parts: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"epoch {epoch}/{args.epochs} [val]", leave=False):
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(float(loss.detach().cpu().item()))
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
                val_pred_parts.append(pred.detach().cpu().numpy())
                val_true_parts.append(yb.detach().cpu().numpy())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        val_acc = float(correct / max(total, 1))

        val_pred_np = np.concatenate(val_pred_parts, axis=0) if val_pred_parts else np.array([], dtype=np.int64)
        val_true_np = np.concatenate(val_true_parts, axis=0) if val_true_parts else np.array([], dtype=np.int64)
        val_macro_f1 = float(
            f1_score(val_true_np, val_pred_np, labels=list(range(num_classes)), average="macro", zero_division=0)
        )
        val_f1_per_class = f1_score(
            val_true_np, val_pred_np, labels=list(range(num_classes)), average=None, zero_division=0
        ).astype(float).tolist()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_f1_per_class"].append(val_f1_per_class)

        labels = list(range(num_classes))
        target_names = splits.label_encoder.inverse_transform(labels).tolist()
        f1_str = " ".join(f"{name}={val_f1_per_class[i]:.3f}" for i, name in enumerate(target_names))
        print(
            f"epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f} | {f1_str}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            bad_epochs = 0
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    if (out_dir / "model.pt").exists():
        model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    model.eval()

    y_pred_parts: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            y_pred_parts.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_pred = np.concatenate(y_pred_parts, axis=0)

    labels = list(range(num_classes))
    target_names = splits.label_encoder.inverse_transform(labels).tolist()

    report = classification_report(y_val, y_pred, labels=labels, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_val, y_pred, labels=labels).tolist()

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "confusion_matrix.json").write_text(json.dumps(cm, indent=2))
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "csv": str(Path(args.csv).as_posix()),
                "seq_len": int(seq_len),
                "num_classes": int(num_classes),
                "arch": args.arch,
                "mlp_hidden": int(args.mlp_hidden),
                "feature_columns": splits.feature_columns,
                "label_order": target_names,
                "scaler_mean": splits.scaler.mean_.astype(float).tolist(),
                "scaler_scale": splits.scaler.scale_.astype(float).tolist(),
                "use_class_weights": bool(args.use_class_weights),
                "use_smote": bool(args.use_smote),
                "seed": int(args.seed),
                "device": str(device),
            },
            indent=2,
        )
    )

    print("Saved artifacts to:", out_dir)
    print("Validation classification report (macro avg f1):", report["macro avg"]["f1-score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

