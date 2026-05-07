from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from lstm_kan.inference import predict_from_csv


ARTIFACT_ROOT = Path("artifacts")
DEFAULT_MODEL = "LSTMKAN"

app = FastAPI(title="LSTM-KAN Energy Forecast API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_artifact_dir(model_name: str) -> Path:
    latest_dir = ARTIFACT_ROOT / "latest" / model_name
    if latest_dir.exists():
        return latest_dir

    candidates = sorted(ARTIFACT_ROOT.glob(f"*/{model_name}"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No trained artifact found for {model_name}. Train a model first with `python train_local.py --copy-latest`."
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    models = []
    latest_root = ARTIFACT_ROOT / "latest"
    if latest_root.exists():
        models.extend(sorted([path.name for path in latest_root.iterdir() if path.is_dir()]))
    return {"models": models}


@app.post("/predict/file")
async def predict_file(
    file: UploadFile = File(...),
    model_name: str = Query(DEFAULT_MODEL),
    batch_size: int = Query(1024, ge=1),
):
    try:
        artifact_dir = _resolve_artifact_dir(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    suffix = Path(file.filename or "input.csv").suffix or ".csv"
    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        result = predict_from_csv(artifact_dir, tmp_path, batch_size=batch_size)
        predictions: pd.DataFrame = result["predictions"]
        preview = predictions.tail(100).copy()
        preview["Datetime"] = preview["Datetime"].astype(str)
        return {
            "model_name": result["model_name"],
            "artifact_dir": result["artifact_dir"],
            "metrics": result["metrics"],
            "rows": preview.to_dict(orient="records"),
            "row_count": len(predictions),
        }
    finally:
        tmp_path.unlink(missing_ok=True)
