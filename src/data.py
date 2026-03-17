from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


LABEL_ORDER = ["N", "SVEB", "VEB", "F", "Q"]


@dataclass(frozen=True)
class DatasetSplits:
    x_train: np.ndarray
    x_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    label_encoder: LabelEncoder
    scaler: StandardScaler
    feature_columns: list[str]


def load_csv_as_sequence(
    csv_path: str | Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> DatasetSplits:
    """
    Loads the dataset and returns (X, y) as a 1D "sequence":
    - X shape: (N, 32, 1)
    - y shape: (N,)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if "type" not in df.columns:
        raise ValueError("Expected a 'type' column with class labels.")

    drop_cols = [c for c in ["record", "type"] if c in df.columns]
    feature_columns = [c for c in df.columns if c not in drop_cols]

    x = df[feature_columns].astype(np.float32).to_numpy()
    y_raw = df["type"].astype(str).to_numpy()

    # Standardize features (tabular), then reshape into (length, channels).
    scaler = StandardScaler()
    x = scaler.fit_transform(x).astype(np.float32)
    x = x[..., np.newaxis]  # (N, 32, 1)

    le = LabelEncoder()
    le.fit(LABEL_ORDER)

    unknown = sorted(set(np.unique(y_raw)) - set(le.classes_))
    if unknown:
        raise ValueError(f"Found unexpected labels in 'type': {unknown}")

    y = le.transform(y_raw).astype(np.int64)

    strat = y if stratify else None
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    return DatasetSplits(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        label_encoder=le,
        scaler=scaler,
        feature_columns=feature_columns,
    )


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Balanced class weights: n_samples / (n_classes * n_samples_in_class).
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n = y.shape[0]
    k = classes.shape[0]
    weights = {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, counts)}
    return weights


def apply_smote(
    x_train: np.ndarray, y_train: np.ndarray, *, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTE expects 2D inputs. We flatten (N, L, C) -> (N, L*C) then reshape back.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SMOTE requires imbalanced-learn. Install requirements.txt."
        ) from e

    if x_train.ndim != 3:
        raise ValueError("Expected x_train to have shape (N, L, C).")
    n, l, c = x_train.shape
    x2 = x_train.reshape(n, l * c)
    smote = SMOTE(random_state=random_state)
    x_res, y_res = smote.fit_resample(x2, y_train)
    x_res = x_res.reshape(x_res.shape[0], l, c).astype(np.float32)
    y_res = np.asarray(y_res).astype(np.int64)
    return x_res, y_res

