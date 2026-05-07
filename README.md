# LSTM-KAN for Time Series Forecasting

This repository explores GRU/LSTM models where the final dense regression head can be replaced with a KAN (Kolmogorov-Arnold Network) layer. The original experiment targeted hourly energy consumption forecasting. The cleaned version keeps the original Kaggle-style notebook as an archive and adds reusable local training, a Colab-ready notebook, and lightweight deployment dashboards.

![Results](images/1.png)

## What Was Improved

The archived notebook still works as historical context, but several parts were not ideal for research reuse:

- The original preprocessing fitted `MinMaxScaler` before the train/test split, which leaks test statistics into training. The new pipeline fits scaling only on the training rows.
- Paths were hard-coded to Kaggle (`/kaggle/...`). The new code works from the repo root, local Jupyter, and Colab.
- The old `evaluate()` returned the `sMAPE` function object instead of the computed metric value.
- Hidden states were carried across shuffled batches. The new training loop treats each sliding-window sample independently.
- The original workflow had no validation split, early stopping, seed control, CLI, saved metadata, or reusable inference path.

## Recommended Notebook Path

Open:

```text
notebooks/colab_lstm_kan_timeseries.ipynb
```

The notebook bootstraps the repository in Colab, trains `LSTMKAN` by default, saves an artifact, evaluates held-out data, and plots predicted vs actual values. For a full comparison, change:

```python
MODEL_NAMES = ["GRU", "GRUKAN", "LSTM", "LSTMKAN"]
```

The old notebook remains available at:

```text
lstm-kan-for-energy-consumption-prediction.ipynb
```

## Local Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Quick smoke training:

```bash
python train_local.py --models LSTMKAN --max-files 1 --epochs 2 --copy-latest
```

Full comparison:

```bash
python train_local.py --models GRU GRUKAN LSTM LSTMKAN --epochs 20 --batch-size 1024 --copy-latest
```

Artifacts are saved under:

```text
artifacts/<run-name>/<MODEL_NAME>/
artifacts/latest/<MODEL_NAME>/
```

Each artifact contains:

- `model.pt`: PyTorch checkpoint.
- `scalers.pkl`: feature and target scalers.
- `metadata.json`: model config, feature columns, window size, history, and dataset metadata.

## Dashboard and Deployment Options

### Streamlit

Run this after training with `--copy-latest`:

```bash
streamlit run streamlit_app.py
```

Upload one of the energy CSV files and inspect MAE, RMSE, sMAPE, and the prediction chart.

### FastAPI + Node.js Dashboard

Start the Python API:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Start the Node.js dashboard:

```bash
node dashboard-node/server.js
```

Open:

```text
http://127.0.0.1:3000
```

The Node dashboard is a lightweight static client that sends uploaded CSV files to the FastAPI backend. It uses the latest trained artifact for the selected model.

## Python API Example

```python
from lstm_kan.inference import predict_from_csv

result = predict_from_csv(
    artifact_dir="artifacts/latest/LSTMKAN",
    csv_path="dataset/DEOK_hourly.csv",
)

print(result["metrics"])
print(result["predictions"].tail())
```

## Dataset Format

Each CSV must contain:

- `Datetime`: hourly timestamp.
- one numeric target column, for example `DEOK_MW`, `DOM_MW`, or `AEP_MW`.

The pipeline creates these input features:

```text
consumption, hour, dayofweek, month, dayofyear
```

## Research Notes

This project is useful as a small reference implementation, not as a final benchmark claim. For publication-grade experiments, run several seeds, report per-dataset metrics, compare against naive seasonal baselines, tune KAN grid settings, and keep all preprocessing decisions fixed before test evaluation.
