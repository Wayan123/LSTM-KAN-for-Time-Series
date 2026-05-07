from __future__ import annotations

import numpy as np


def _to_1d(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def mae(y_true, y_pred) -> float:
    true = _to_1d(y_true)
    pred = _to_1d(y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true, y_pred) -> float:
    true = _to_1d(y_true)
    pred = _to_1d(y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    true = _to_1d(y_true)
    pred = _to_1d(y_pred)
    denom = np.abs(true) + np.abs(pred)
    diff = np.abs(pred - true)
    score = np.zeros_like(denom)
    mask = denom > eps
    score[mask] = 2.0 * diff[mask] / denom[mask]
    return float(np.mean(score) * 100.0)
