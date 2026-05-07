from __future__ import annotations

import importlib.util

import numpy as np

from lstm_kan.data import load_energy_dataset
from lstm_kan.metrics import smape
from lstm_kan.scalers import SimpleMinMaxScaler


def test_simple_minmax_scaler_roundtrip():
    scaler = SimpleMinMaxScaler()
    values = np.array([[2.0], [4.0], [8.0]])
    scaled = scaler.fit_transform(values)
    restored = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(restored, values)


def test_load_energy_dataset_shapes():
    dataset = load_energy_dataset("dataset", window_size=12, max_files=1, train_ratio=0.7, val_ratio=0.15)
    assert dataset.train_x.ndim == 3
    assert dataset.train_x.shape[1:] == (12, 5)
    assert dataset.train_y.shape[1] == 1
    assert dataset.val_x.shape[1:] == (12, 5)
    assert dataset.feature_columns == ["consumption", "hour", "dayofweek", "month", "dayofyear"]


def test_target_labels_inverse_transform_to_mw_scale():
    dataset = load_energy_dataset("dataset", window_size=12, max_files=1, train_ratio=0.7, val_ratio=0.15)
    restored = dataset.target_scaler.inverse_transform(dataset.train_y[:100])
    assert restored.mean() > 1000.0


def test_smape_zero_denominator_is_stable():
    assert smape([0.0, 1.0], [0.0, 1.0]) == 0.0


def test_lstm_kan_forward_shape_when_torch_available():
    if importlib.util.find_spec("torch") is None:
        return

    import torch
    from lstm_kan.models import build_model

    model = build_model("LSTMKAN", input_dim=5, hidden_dim=8, n_layers=1, dropout=0.0)
    out, _ = model(torch.rand(4, 12, 5))
    assert tuple(out.shape) == (4, 1)
