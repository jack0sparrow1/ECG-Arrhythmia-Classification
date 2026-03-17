from __future__ import annotations

import torch
from torch import nn


class MlpClassifier(nn.Module):
    """
    Baseline for engineered features.
    Input: (B, L, 1) -> flatten to (B, L)
    """

    def __init__(
        self,
        *,
        seq_len: int,
        num_classes: int,
        hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CnnBiLstm1D(nn.Module):
    """
    Input: (B, L, 1)
    - Conv1D expects (B, C, L), so we transpose.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        num_classes: int,
        conv_filters: int = 64,
        lstm_units: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(1, conv_filters, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_filters)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # After two pools with stride 2, length becomes floor(L/4).
        lstm_input_size = conv_filters
        self.bilstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * lstm_units, num_classes)

        self.seq_len = seq_len
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1) -> (B, 1, L)
        x = x.transpose(1, 2)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)  # (B, F, L')

        x = x.transpose(1, 2)  # (B, L', F) for LSTM
        x, _ = self.bilstm(x)  # (B, L', 2H)
        x = x[:, -1, :]  # last timestep
        x = self.drop(x)
        logits = self.fc(x)
        return logits


def build_cnn_bilstm(
    *, seq_len: int, num_classes: int, conv_filters: int = 64, lstm_units: int = 64, dropout: float = 0.3
) -> CnnBiLstm1D:
    return CnnBiLstm1D(
        seq_len=seq_len,
        num_classes=num_classes,
        conv_filters=conv_filters,
        lstm_units=lstm_units,
        dropout=dropout,
    )


def build_mlp(*, seq_len: int, num_classes: int, hidden: int = 128, dropout: float = 0.3) -> MlpClassifier:
    return MlpClassifier(seq_len=seq_len, num_classes=num_classes, hidden=hidden, dropout=dropout)

