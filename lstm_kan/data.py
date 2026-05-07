from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .scalers import SimpleMinMaxScaler


TARGET_ALIAS = "consumption"
DEFAULT_FEATURE_COLUMNS = [TARGET_ALIAS, "hour", "dayofweek", "month", "dayofyear"]


@dataclass
class FileSummary:
    file_name: str
    total_rows: int
    train_rows: int
    val_rows: int
    test_rows: int


@dataclass
class PreparedDataset:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x_by_file: dict[str, np.ndarray]
    test_y_by_file: dict[str, np.ndarray]
    feature_scaler: SimpleMinMaxScaler
    target_scaler: SimpleMinMaxScaler
    feature_columns: list[str]
    target_column: str
    file_summaries: list[FileSummary]


def discover_csv_files(data_dir: str | Path, file_names: Iterable[str] | None = None, max_files: int | None = None) -> list[Path]:
    data_path = Path(data_dir)
    if file_names is not None:
        files = [data_path / name for name in file_names]
    else:
        files = sorted(data_path.glob("*.csv"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")
    return files


def infer_target_column(df: pd.DataFrame) -> str:
    candidates = [col for col in df.columns if col != "Datetime"]
    if not candidates:
        raise ValueError("CSV file must include one target column besides Datetime.")
    numeric = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]
    if numeric:
        return numeric[0]
    return candidates[0]


def enrich_energy_frame(
    df: pd.DataFrame,
    target_column: str | None = None,
    *,
    rename_target: bool = True,
) -> tuple[pd.DataFrame, str]:
    if "Datetime" not in df.columns:
        raise ValueError("Expected a Datetime column in the energy CSV file.")

    frame = df.copy()
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], errors="coerce")
    if frame["Datetime"].isna().any():
        raise ValueError("Failed to parse one or more Datetime values.")
    frame = frame.sort_values("Datetime").reset_index(drop=True)

    if target_column is not None and target_column not in frame.columns:
        target_column = None
    if target_column is None:
        target_column = infer_target_column(frame)

    target_name = TARGET_ALIAS if rename_target else target_column
    frame[target_name] = pd.to_numeric(frame[target_column], errors="coerce")
    if frame[target_name].isna().any():
        raise ValueError(f"Target column {target_column!r} contains non-numeric values.")

    dt = frame["Datetime"].dt
    frame["hour"] = dt.hour.astype(np.float32)
    frame["dayofweek"] = dt.dayofweek.astype(np.float32)
    frame["month"] = dt.month.astype(np.float32)
    frame["dayofyear"] = dt.dayofyear.astype(np.float32)

    keep = ["Datetime", target_name, "hour", "dayofweek", "month", "dayofyear"]
    frame = frame[keep]
    return frame, target_name


def split_row_counts(total_rows: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    train_rows = max(1, int(round(total_rows * train_ratio)))
    val_rows = max(1, int(round(total_rows * val_ratio)))
    if train_rows + val_rows >= total_rows:
        val_rows = max(1, total_rows - train_rows - 1)
    test_rows = total_rows - train_rows - val_rows
    if test_rows <= 0:
        raise ValueError("Not enough rows left for the test split.")
    return train_rows, val_rows, test_rows


def build_windows(values: np.ndarray, targets: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if len(values) <= window_size:
        raise ValueError("window_size is larger than the available rows.")

    windows = []
    labels = []
    positions = []
    for label_idx in range(window_size, len(values)):
        windows.append(values[label_idx - window_size : label_idx])
        labels.append(targets[label_idx])
        positions.append(label_idx)
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.float32).reshape(-1, 1),
        np.asarray(positions, dtype=np.int64),
    )


def _empty_windows(window_size: int, feature_count: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0, window_size, feature_count), dtype=np.float32),
        np.empty((0, 1), dtype=np.float32),
    )


def _concat_windows(items: list[np.ndarray], fallback_shape: tuple[int, ...]) -> np.ndarray:
    if not items:
        return np.empty(fallback_shape, dtype=np.float32)
    return np.concatenate(items, axis=0).astype(np.float32)


def load_energy_dataset(
    data_dir: str | Path,
    *,
    window_size: int = 90,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_files: int | None = None,
    file_names: Iterable[str] | None = None,
) -> PreparedDataset:
    files = discover_csv_files(data_dir, file_names=file_names, max_files=max_files)

    raw_frames: list[tuple[str, pd.DataFrame, str, int, int, int]] = []
    training_rows: list[pd.DataFrame] = []
    feature_columns = DEFAULT_FEATURE_COLUMNS.copy()

    for csv_path in files:
        raw_df = pd.read_csv(csv_path)
        frame, target_column = enrich_energy_frame(raw_df)
        train_rows, val_rows, test_rows = split_row_counts(len(frame), train_ratio, val_ratio)
        raw_frames.append((csv_path.name, frame, target_column, train_rows, val_rows, test_rows))
        training_rows.append(frame.iloc[:train_rows].copy())

    combined_train = pd.concat(training_rows, ignore_index=True)
    feature_scaler = SimpleMinMaxScaler()
    target_scaler = SimpleMinMaxScaler()
    feature_scaler.fit(combined_train[feature_columns])
    target_scaler.fit(combined_train[[TARGET_ALIAS]])

    train_x_parts: list[np.ndarray] = []
    train_y_parts: list[np.ndarray] = []
    val_x_parts: list[np.ndarray] = []
    val_y_parts: list[np.ndarray] = []
    test_x_by_file: dict[str, np.ndarray] = {}
    test_y_by_file: dict[str, np.ndarray] = {}
    file_summaries: list[FileSummary] = []

    for file_name, frame, target_column, train_rows, val_rows, test_rows in raw_frames:
        transformed = frame.copy()
        transformed[feature_columns] = feature_scaler.transform(transformed[feature_columns])

        values = transformed[feature_columns].to_numpy(dtype=np.float32)
        targets = target_scaler.transform(frame[[TARGET_ALIAS]]).reshape(-1).astype(np.float32)
        windows, labels, positions = build_windows(values, targets, window_size)

        train_mask = positions < train_rows
        val_mask = (positions >= train_rows) & (positions < train_rows + val_rows)
        test_mask = positions >= train_rows + val_rows

        if train_mask.any():
            train_x_parts.append(windows[train_mask])
            train_y_parts.append(labels[train_mask])
        if val_mask.any():
            val_x_parts.append(windows[val_mask])
            val_y_parts.append(labels[val_mask])
        if test_mask.any():
            test_x_by_file[file_name] = windows[test_mask].astype(np.float32)
            test_y_by_file[file_name] = labels[test_mask].astype(np.float32)
        else:
            test_x_by_file[file_name], test_y_by_file[file_name] = _empty_windows(window_size, len(feature_columns))

        file_summaries.append(
            FileSummary(
                file_name=file_name,
                total_rows=len(frame),
                train_rows=train_rows,
                val_rows=val_rows,
                test_rows=test_rows,
            )
        )

    train_x = _concat_windows(train_x_parts, (0, window_size, len(feature_columns)))
    train_y = _concat_windows(train_y_parts, (0, 1))
    val_x = _concat_windows(val_x_parts, (0, window_size, len(feature_columns)))
    val_y = _concat_windows(val_y_parts, (0, 1))

    return PreparedDataset(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x_by_file=test_x_by_file,
        test_y_by_file=test_y_by_file,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=feature_columns,
        target_column=TARGET_ALIAS,
        file_summaries=file_summaries,
    )
