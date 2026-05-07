from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import TARGET_ALIAS, build_windows, enrich_energy_frame
from .metrics import mae, rmse, smape
from .models import build_model


@dataclass
class LoadedArtifact:
    model: torch.nn.Module
    feature_scaler: object
    target_scaler: object
    metadata: dict
    artifact_dir: Path


def load_artifact(artifact_dir: str | Path, device: str | torch.device = "cpu") -> LoadedArtifact:
    artifact_path = Path(artifact_dir)
    with open(artifact_path / "metadata.json", "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    with open(artifact_path / "scalers.pkl", "rb") as fh:
        scalers = pickle.load(fh)
    checkpoint = torch.load(artifact_path / "model.pt", map_location=device)

    model = build_model(
        checkpoint["model_name"],
        input_dim=len(metadata["feature_columns"]),
        hidden_dim=checkpoint["model_kwargs"]["hidden_dim"],
        output_dim=checkpoint["model_kwargs"].get("output_dim", 1),
        n_layers=checkpoint["model_kwargs"]["n_layers"],
        dropout=checkpoint["model_kwargs"].get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return LoadedArtifact(
        model=model,
        feature_scaler=scalers["feature_scaler"],
        target_scaler=scalers["target_scaler"],
        metadata=metadata,
        artifact_dir=artifact_path,
    )


def _predict_windows(
    model: torch.nn.Module,
    windows: np.ndarray,
    device: str | torch.device = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    device = torch.device(device)
    if len(windows) == 0:
        return np.empty((0, 1), dtype=np.float32)

    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(device).float()
            out, _ = model(batch)
            preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_frame(
    artifact: LoadedArtifact,
    frame: pd.DataFrame,
    *,
    batch_size: int = 1024,
) -> pd.DataFrame:
    window_size = artifact.metadata["window_size"]
    feature_columns = artifact.metadata["feature_columns"]

    enriched, _ = enrich_energy_frame(frame)
    transformed = enriched.copy()
    transformed[feature_columns] = artifact.feature_scaler.transform(transformed[feature_columns])

    values = transformed[feature_columns].to_numpy(dtype=np.float32)
    targets = artifact.target_scaler.transform(enriched[[TARGET_ALIAS]]).reshape(-1).astype(np.float32)
    windows, labels, positions = build_windows(values, targets, window_size)
    pred_scaled = _predict_windows(artifact.model, windows, device=next(artifact.model.parameters()).device, batch_size=batch_size)

    predicted = artifact.target_scaler.inverse_transform(pred_scaled).reshape(-1)
    actual = artifact.target_scaler.inverse_transform(labels).reshape(-1)
    timestamps = enriched["Datetime"].iloc[positions].reset_index(drop=True)

    result = pd.DataFrame(
        {
            "Datetime": timestamps,
            "actual": actual,
            "predicted": predicted,
        }
    )
    result["abs_error"] = np.abs(result["actual"] - result["predicted"])
    return result


def predict_from_csv(
    artifact_dir: str | Path,
    csv_path: str | Path,
    *,
    batch_size: int = 1024,
    device: str | torch.device = "cpu",
) -> dict:
    artifact = load_artifact(artifact_dir, device=device)
    frame = pd.read_csv(csv_path)
    predictions = predict_frame(artifact, frame, batch_size=batch_size)
    metrics = {
        "mae": mae(predictions["actual"], predictions["predicted"]),
        "rmse": rmse(predictions["actual"], predictions["predicted"]),
        "smape": smape(predictions["actual"], predictions["predicted"]),
    }
    return {
        "artifact_dir": str(artifact.artifact_dir),
        "model_name": artifact.metadata["model_name"],
        "metrics": metrics,
        "predictions": predictions,
    }
