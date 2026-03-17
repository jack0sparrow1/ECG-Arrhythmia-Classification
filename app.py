from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.explain import gradcam_1d
from src.model import build_cnn_bilstm, build_mlp


def _load_run(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {run_dir}")
    meta = json.loads(meta_path.read_text())
    return meta


def _resolve_csv(meta: dict) -> Path:
    csv_path = Path(meta["csv"])
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return csv_path


def _build_model(meta: dict, device: torch.device) -> torch.nn.Module:
    seq_len = int(meta["seq_len"])
    num_classes = int(meta["num_classes"])
    arch = meta.get("arch", "cnn_bilstm")

    if arch == "mlp":
        model = build_mlp(seq_len=seq_len, num_classes=num_classes, hidden=int(meta.get("mlp_hidden", 128)))
    else:
        model = build_cnn_bilstm(seq_len=seq_len, num_classes=num_classes)

    model_path = Path(meta.get("model_path", ""))  # unused, but kept for future
    state_path = Path(st.session_state.get("state_path", "")) if "state_path" in st.session_state else None
    # default file name
    if state_path is None or not state_path.exists():
        state_path = None
    _ = model_path  # silence linters

    return model.to(device)


def _load_state_dict(model: torch.nn.Module, run_dir: Path, device: torch.device) -> None:
    state_path = run_dir / "model.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing model.pt in {run_dir}")
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()


def _standardize(x: np.ndarray, meta: dict) -> np.ndarray:
    mean = np.asarray(meta["scaler_mean"], dtype=np.float32)
    scale = np.asarray(meta["scaler_scale"], dtype=np.float32)
    return ((x - mean) / (scale + 1e-12)).astype(np.float32)


def _ensure_scaler_in_meta(meta: dict, csv_path: Path) -> dict:
    """
    Backward-compatible: older runs may not have scaler params saved.
    In that case, fit a scaler on the full CSV features so the app can run.
    (Best practice: retrain once so meta.json includes scaler_mean/scale.)
    """
    if "scaler_mean" in meta and "scaler_scale" in meta:
        return meta

    feature_cols = list(meta["feature_columns"])
    df = pd.read_csv(csv_path)
    x = df[feature_cols].astype(np.float32).to_numpy()
    mean = x.mean(axis=0).astype(np.float32)
    scale = x.std(axis=0).astype(np.float32)
    scale[scale == 0] = 1.0

    meta = dict(meta)
    meta["scaler_mean"] = mean.astype(float).tolist()
    meta["scaler_scale"] = scale.astype(float).tolist()
    meta["_scaler_fitted_in_app"] = True
    return meta


def _load_meta_and_csv(run_dir: Path) -> tuple[dict, Path]:
    meta = _load_run(run_dir)
    csv_path = _resolve_csv(meta)
    meta = _ensure_scaler_in_meta(meta, csv_path)
    return meta, csv_path


def _parse_manual_vector(text: str, expected_len: int) -> np.ndarray:
    parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
    vals = [float(p) for p in parts]
    if len(vals) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(vals)}.")
    return np.asarray(vals, dtype=np.float32)


def _predict(
    model: torch.nn.Module,
    x_seq: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(x_seq[None, :, None]).float().to(device))
        probs = torch.softmax(logits[0], dim=0).cpu().numpy().astype(np.float32)
    return probs


def _gradcam_plot(
    model: torch.nn.Module,
    x_seq_std: np.ndarray,
    *,
    pred_idx: int,
) -> plt.Figure:
    cam = gradcam_1d(model, x_seq_std[:, None], conv_layer_name="conv2", class_index=pred_idx)
    cam_up = np.interp(np.linspace(0, len(cam) - 1, num=len(x_seq_std)), np.arange(len(cam)), cam)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    xs = np.arange(len(x_seq_std))
    ax.plot(xs, x_seq_std, lw=1.5, label="standardized input", color="#1f77b4")
    ax.set_xlabel("feature position (0..31)")

    # Make Grad-CAM unambiguous: show it on a second axis (0..1) in red + shaded.
    ax2 = ax.twinx()
    ax2.plot(xs, cam_up, color="#d62728", lw=2.0, label="Grad-CAM (0..1)")
    ax2.fill_between(xs, 0.0, cam_up, color="#d62728", alpha=0.25)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Grad-CAM")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.tight_layout()
    return fig


def _bar_probs(probs: np.ndarray, labels: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(labels, probs, color="#4c78a8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("probability")
    fig.tight_layout()
    return fig


def _get_input_from_csv(df: pd.DataFrame, feature_cols: list[str], idx: int) -> np.ndarray:
    row = df.iloc[int(idx)]
    x = row[feature_cols].astype(np.float32).to_numpy()
    return x


st.set_page_config(page_title="ECG Arrhythmia Inference", layout="wide")
st.title("ECG Arrhythmia Classification — Interactive Inference")

st.sidebar.header("Models")
ui_mode = st.sidebar.radio("Mode", options=["Single model", "Compare (MLP vs CNN)"])

run_dir_single = st.sidebar.text_input("Run directory", value="artifacts/compare/cnn_bilstm")
run_dir_mlp = st.sidebar.text_input("MLP run directory", value="artifacts/compare/mlp")
run_dir_cnn = st.sidebar.text_input("CNN+BiLSTM run directory", value="artifacts/compare/cnn_bilstm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"Device: **{device}**")

@st.cache_resource
def _cached_model(run_dir_abs: str, meta_json: str, device_str: str) -> torch.nn.Module:
    _ = meta_json  # cache bust when meta changes
    dev = torch.device(device_str)
    m = _build_model(json.loads(meta_json), dev)
    _load_state_dict(m, Path(run_dir_abs), dev)
    return m


def _load_for_run_dir(run_dir: Path) -> tuple[dict, Path, torch.nn.Module]:
    meta, csv_path = _load_meta_and_csv(run_dir)
    model = _cached_model(str(run_dir.resolve()), json.dumps(meta, sort_keys=True), str(device))
    return meta, csv_path, model


try:
    if ui_mode == "Single model":
        meta_a, csv_a, model_a = _load_for_run_dir(Path(run_dir_single))
        runs = [("Model", meta_a, csv_a, model_a)]
    else:
        meta_m, csv_m, model_m = _load_for_run_dir(Path(run_dir_mlp))
        meta_c, csv_c, model_c = _load_for_run_dir(Path(run_dir_cnn))
        runs = [("MLP", meta_m, csv_m, model_m), ("CNN+BiLSTM", meta_c, csv_c, model_c)]
except Exception as e:
    st.error(str(e))
    st.stop()

# Sidebar info + compatibility warnings.
for title, meta_i, csv_i, _model_i in runs:
    arch_i = meta_i.get("arch", "cnn_bilstm")
    st.sidebar.write(f"{title}: **{arch_i}**")
    st.sidebar.write(f"{title} CSV: `{csv_i}`")
    if meta_i.get("_scaler_fitted_in_app"):
        st.sidebar.warning(f"{title}: scaler params missing in meta.json; fitted from CSV.")

# Use the first run for feature ordering + row selection.
meta0, csv0 = runs[0][1], runs[0][2]
feature_cols0 = list(meta0["feature_columns"])

st.subheader("Choose an input")
mode = st.radio("Input mode", options=["Pick a row from CSV", "Paste 32 feature values"], horizontal=True)

x_raw: Optional[np.ndarray] = None
y_true: Optional[str] = None

if mode == "Pick a row from CSV":
    df = pd.read_csv(csv0)
    st.write(f"Rows: **{len(df):,}**  |  Columns: **{len(df.columns)}**")
    idx = st.number_input("Row index", min_value=0, max_value=int(len(df) - 1), value=0, step=1)
    x_raw = _get_input_from_csv(df, feature_cols0, int(idx))
    if "type" in df.columns:
        y_true = str(df.iloc[int(idx)]["type"])

    with st.expander("Show raw row (selected columns)"):
        preview = pd.DataFrame([x_raw], columns=feature_cols0)
        st.dataframe(preview, use_container_width=True)

else:
    st.caption("Paste 32 comma-separated numbers in the same order as `feature_columns` saved in the run.")
    st.code(", ".join(feature_cols0), language="text")
    txt = st.text_area("32 values", height=120, value="")
    if txt.strip():
        try:
            x_raw = _parse_manual_vector(txt, expected_len=len(feature_cols0))
        except Exception as e:
            st.error(str(e))

if x_raw is None:
    st.stop()

st.subheader("Results")

cols = st.columns(len(runs), gap="large")
for col, (title, meta_i, _csv_i, model_i) in zip(cols, runs):
    labels_i = list(meta_i["label_order"])
    arch_i = meta_i.get("arch", "cnn_bilstm")

    x_std_i = _standardize(x_raw, meta_i)
    probs_i = _predict(model_i, x_std_i, device=device)
    pred_idx_i = int(np.argmax(probs_i))
    pred_label_i = labels_i[pred_idx_i]

    with col:
        st.markdown(f"### {title}")
        if y_true is not None:
            st.write(f"True label: **{y_true}**")
        st.write(f"Predicted: **{pred_label_i}**  (p={probs_i[pred_idx_i]:.3f})")
        st.pyplot(_bar_probs(probs_i, labels_i), width="stretch")

        if arch_i == "cnn_bilstm":
            st.write("Grad-CAM")
            fig = _gradcam_plot(model_i, x_std_i, pred_idx=pred_idx_i)
            st.pyplot(fig, width="stretch")
        else:
            st.caption("Grad-CAM not available for MLP.")

