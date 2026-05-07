from __future__ import annotations

import json
import random
import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import PreparedDataset
from .metrics import mae, rmse, smape


@dataclass
class TrainingConfig:
    model_name: str
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 1024
    patience: int = 5
    min_delta: float = 0.0
    device: str = "cpu"


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]
    train_mae: list[float]
    val_mae: list[float]
    train_rmse: list[float]
    val_rmse: list[float]
    train_smape: list[float]
    val_smape: list[float]
    best_epoch: int
    best_val_loss: float


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).float()

        if training:
            optimizer.zero_grad()

        out, _ = model(x)
        loss = criterion(out, y)

        if training:
            loss.backward()
            optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        preds.append(out.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())

    if len(loader.dataset) == 0:
        return float("nan"), np.empty((0, 1)), np.empty((0, 1))

    return (
        total_loss / len(loader.dataset),
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
    )


def predict_windows(
    model: nn.Module,
    windows: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    device = torch.device(device)
    if len(windows) == 0:
        return np.empty((0, 1), dtype=np.float32)

    model.to(device)
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(device).float()
            out, _ = model(batch)
            preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_by_file(
    model: nn.Module,
    x_by_file: dict[str, np.ndarray],
    y_by_file: dict[str, np.ndarray],
    *,
    target_scaler,
    device: str | torch.device = "cpu",
    batch_size: int = 1024,
) -> list[dict]:
    rows = []
    for file_name, x in x_by_file.items():
        y = y_by_file[file_name]
        if len(x) == 0:
            continue
        pred_scaled = predict_windows(model, x, device=device, batch_size=batch_size)
        pred = target_scaler.inverse_transform(pred_scaled)
        target = target_scaler.inverse_transform(y)
        rows.append(
            {
                "file": file_name,
                "rows": int(len(x)),
                "mae": mae(target, pred),
                "rmse": rmse(target, pred),
                "smape": smape(target, pred),
            }
        )
    return rows


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    learning_rate: float,
    epochs: int,
    patience: int,
    device: str | torch.device = "cpu",
    min_delta: float = 0.0,
    target_scaler=None,
    verbose: bool = True,
) -> tuple[nn.Module, TrainingHistory]:
    device = torch.device(device)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = TrainingHistory(
        train_loss=[],
        val_loss=[],
        train_mae=[],
        val_mae=[],
        train_rmse=[],
        val_rmse=[],
        train_smape=[],
        val_smape=[],
        best_epoch=0,
        best_val_loss=float("inf"),
    )

    best_state = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        train_loss, train_pred, train_true = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_pred, val_true = _run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )

        if target_scaler is not None and len(train_pred):
            train_pred = target_scaler.inverse_transform(train_pred)
            train_true = target_scaler.inverse_transform(train_true)
        if target_scaler is not None and len(val_pred):
            val_pred = target_scaler.inverse_transform(val_pred)
            val_true = target_scaler.inverse_transform(val_true)

        train_mae = mae(train_true, train_pred) if len(train_pred) else float("nan")
        val_mae = mae(val_true, val_pred) if len(val_pred) else float("nan")
        train_rmse = rmse(train_true, train_pred) if len(train_pred) else float("nan")
        val_rmse = rmse(val_true, val_pred) if len(val_pred) else float("nan")
        train_smape = smape(train_true, train_pred) if len(train_pred) else float("nan")
        val_smape = smape(val_true, val_pred) if len(val_pred) else float("nan")

        history.train_loss.append(float(train_loss))
        history.val_loss.append(float(val_loss))
        history.train_mae.append(float(train_mae))
        history.val_mae.append(float(val_mae))
        history.train_rmse.append(float(train_rmse))
        history.val_rmse.append(float(val_rmse))
        history.train_smape.append(float(train_smape))
        history.val_smape.append(float(val_smape))

        if verbose:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"val_smape={val_smape:.3f}%"
            )

        if val_loss + min_delta < history.best_val_loss:
            history.best_val_loss = float(val_loss)
            history.best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"Early stopping at epoch {epoch}. Best epoch: {history.best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def save_artifact(
    model: nn.Module,
    artifact_dir: str | Path,
    *,
    model_name: str,
    model_kwargs: dict,
    feature_columns: Iterable[str],
    target_column: str,
    window_size: int,
    feature_scaler,
    target_scaler,
    history: TrainingHistory,
    extra_metadata: dict | None = None,
) -> Path:
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "state_dict": model.state_dict(),
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
    }
    torch.save(checkpoint, artifact_path / "model.pt")

    with open(artifact_path / "scalers.pkl", "wb") as fh:
        pickle.dump({"feature_scaler": feature_scaler, "target_scaler": target_scaler}, fh)

    metadata = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "window_size": window_size,
        "history": {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "train_mae": history.train_mae,
            "val_mae": history.val_mae,
            "train_rmse": history.train_rmse,
            "val_rmse": history.val_rmse,
            "train_smape": history.train_smape,
            "val_smape": history.val_smape,
        },
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(artifact_path / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return artifact_path


def build_loaders(dataset: PreparedDataset, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_loader = build_dataloader(dataset.train_x, dataset.train_y, batch_size, shuffle=True)
    val_loader = build_dataloader(dataset.val_x, dataset.val_y, batch_size, shuffle=False)
    return train_loader, val_loader
