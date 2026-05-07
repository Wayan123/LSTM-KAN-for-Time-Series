from __future__ import annotations

import numpy as np


class SimpleMinMaxScaler:
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.min_: np.ndarray | None = None

    def fit(self, values):
        arr = self._as_2d(values)
        data_min = arr.min(axis=0)
        data_max = arr.max(axis=0)
        data_range = data_max - data_min
        safe_range = np.where(data_range == 0.0, 1.0, data_range)
        feature_min, feature_max = self.feature_range

        self.data_min_ = data_min
        self.data_max_ = data_max
        self.scale_ = (feature_max - feature_min) / safe_range
        self.min_ = feature_min - data_min * self.scale_
        return self

    def transform(self, values):
        self._check_fitted()
        arr = self._as_2d(values)
        return arr * self.scale_ + self.min_

    def inverse_transform(self, values):
        self._check_fitted()
        arr = self._as_2d(values)
        return (arr - self.min_) / self.scale_

    def fit_transform(self, values):
        return self.fit(values).transform(values)

    @staticmethod
    def _as_2d(values) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _check_fitted(self):
        if self.scale_ is None or self.min_ is None:
            raise RuntimeError("SimpleMinMaxScaler has not been fitted.")
