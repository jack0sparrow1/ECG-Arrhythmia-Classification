from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import load_csv_as_sequence
from .model import build_cnn_bilstm


def gradcam_1d(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    conv_layer_name: str = "conv2",
    class_index: int | None = None,
) -> np.ndarray:
    """
    1D Grad-CAM over the position axis of a Conv1D layer.
    Returns a 1D heatmap aligned to that layer's output length (after pooling).
    """
    layer = getattr(model, conv_layer_name, None)
    if layer is None:
        raise ValueError(f"Model has no layer attribute named '{conv_layer_name}'")

    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None

    def fwd_hook(_module, _inp, out):
        nonlocal activations
        activations = out

    def bwd_hook(_module, _grad_inp, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    x_t = torch.from_numpy(x[None, ...]).float().to(device)
    x_t.requires_grad_(True)
    logits = model(x_t)
    if class_index is None:
        class_index = int(torch.argmax(logits[0]).item())
    score = logits[:, class_index].sum()
    score.backward()

    h1.remove()
    h2.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Failed to capture activations/gradients for Grad-CAM.")

    # activations/grads: (B, C, L)
    a = activations.detach()[0]
    g = gradients.detach()[0]
    weights = g.mean(dim=1)  # (C,)
    cam = (a * weights[:, None]).sum(dim=0)  # (L,)
    cam = torch.relu(cam)
    cam_np = cam.cpu().numpy().astype(np.float32)
    if float(cam_np.max()) > 0:
        cam_np = cam_np / (cam_np.max() + 1e-8)
    return cam_np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True, help="Training artifact directory")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--conv-layer", type=str, default="conv2")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    if meta.get("arch") not in (None, "cnn_bilstm"):
        raise ValueError("Grad-CAM is only supported for --arch cnn_bilstm (needs Conv1D layers).")
    csv_meta = meta["csv"]
    # meta["csv"] may be absolute or relative. Resolve relative to current workspace.
    csv_path = Path(csv_meta)
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    splits = load_csv_as_sequence(str(csv_path), random_state=meta.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cnn_bilstm(seq_len=int(meta["seq_len"]), num_classes=int(meta["num_classes"])).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()

    x_val, y_val = splits.x_val, splits.y_val
    idxs = np.random.default_rng(123).choice(len(x_val), size=min(args.num_samples, len(x_val)), replace=False)

    out_dir = run_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(idxs):
        x = x_val[idx]
        y_true = int(y_val[idx])
        with torch.no_grad():
            logits = model(torch.from_numpy(x[None, ...]).float().to(device))
            y_pred = int(torch.argmax(logits[0]).item())

        cam = gradcam_1d(model, x, conv_layer_name=args.conv_layer, class_index=y_pred)

        # Upsample CAM to original input length for a simple overlay.
        cam_up = np.interp(
            np.linspace(0, len(cam) - 1, num=x.shape[0]),
            np.arange(len(cam)),
            cam,
        )

        plt.figure(figsize=(10, 3))
        plt.plot(x[:, 0], label="input (standardized features)")
        plt.fill_between(np.arange(x.shape[0]), x[:, 0].min(), x[:, 0].max(), alpha=0.25 * cam_up, label="Grad-CAM")
        plt.title(
            f"idx={idx} true={splits.label_encoder.inverse_transform([y_true])[0]} "
            f"pred={splits.label_encoder.inverse_transform([y_pred])[0]}"
        )
        plt.xlabel("feature position (0..31)")
        plt.tight_layout()
        plt.savefig(out_dir / f"sample_{i}_idx_{idx}.png", dpi=160)
        plt.close()

    print("Saved Grad-CAM plots to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

