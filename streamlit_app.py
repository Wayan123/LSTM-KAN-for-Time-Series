from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from lstm_kan.inference import load_artifact, predict_from_csv


st.set_page_config(page_title="LSTM-KAN Dashboard", layout="wide")
st.title("LSTM-KAN Time Series Dashboard")

artifact_root = Path("artifacts")
latest_root = artifact_root / "latest"
available_models = sorted([path.name for path in latest_root.iterdir() if path.is_dir()]) if latest_root.exists() else []

with st.sidebar:
    st.header("Run Settings")
    model_name = st.selectbox("Model", available_models or ["LSTMKAN"], index=0)
    batch_size = st.number_input("Batch size", min_value=1, value=1024, step=1)
    uploaded = st.file_uploader("Upload an energy CSV", type=["csv"])

artifact_dir = latest_root / model_name if latest_root.exists() else None

if artifact_dir and artifact_dir.exists():
    artifact = load_artifact(artifact_dir)
    st.caption(f"Loaded artifact from {artifact_dir}")
    st.write(
        {
            "feature_columns": artifact.metadata["feature_columns"],
            "window_size": artifact.metadata["window_size"],
            "best_epoch": artifact.metadata.get("best_epoch"),
            "best_val_loss": artifact.metadata.get("best_val_loss"),
        }
    )
else:
    st.warning("Train a model first with `python train_local.py --copy-latest`.")

if uploaded and artifact_dir and artifact_dir.exists():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    try:
        result = predict_from_csv(artifact_dir, tmp_path, batch_size=int(batch_size))
        metrics = result["metrics"]
        predictions: pd.DataFrame = result["predictions"]

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{metrics['mae']:.3f}")
        c2.metric("RMSE", f"{metrics['rmse']:.3f}")
        c3.metric("sMAPE", f"{metrics['smape']:.3f}%")

        st.line_chart(
            predictions.set_index("Datetime")[["actual", "predicted"]],
            height=420,
        )
        st.dataframe(predictions.tail(200), use_container_width=True)
    finally:
        tmp_path.unlink(missing_ok=True)
else:
    st.info("Upload a CSV to generate predictions and compare the actual vs predicted curve.")
