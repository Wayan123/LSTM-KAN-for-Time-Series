# Deployment Guide

This guide describes the supported local deployment options. All options expect that a model has already been trained and saved with `--copy-latest`.

## Prepare an Artifact

Run a small training job first:

```bash
python train_local.py \
  --models LSTMKAN \
  --max-files 1 \
  --epochs 2 \
  --copy-latest
```

This creates:

```text
artifacts/latest/LSTMKAN/model.pt
artifacts/latest/LSTMKAN/scalers.pkl
artifacts/latest/LSTMKAN/metadata.json
```

The inference code needs all three files.

## Option 1: Streamlit

Streamlit is the simplest dashboard option.

```bash
streamlit run streamlit_app.py
```

Use it when you want to upload a CSV and quickly inspect:

- MAE.
- RMSE.
- sMAPE.
- Actual vs predicted line chart.
- Recent prediction rows.

## Option 2: FastAPI

FastAPI is useful when another application needs to call the trained model.

Start the server:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

List latest model artifacts:

```bash
curl http://127.0.0.1:8000/models
```

Upload a CSV for prediction:

```bash
curl -X POST \
  "http://127.0.0.1:8000/predict/file?model_name=LSTMKAN&batch_size=1024" \
  -F "file=@dataset/DEOK_hourly.csv"
```

The response includes:

- `model_name`
- `artifact_dir`
- `metrics`
- `rows`: the last 100 prediction rows
- `row_count`

## Option 3: FastAPI + Node.js Dashboard

The Node.js dashboard is a static client that talks to the FastAPI service.

Terminal 1:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
node dashboard-node/server.js
```

Open:

```text
http://127.0.0.1:3000
```

The dashboard uploads CSV files to:

```text
POST http://127.0.0.1:8000/predict/file
```

## Production Notes

This repository is designed for local experimentation. If you adapt it for production:

- Pin exact dependency versions.
- Restrict CORS origins in `api.py`.
- Add authentication if the API is exposed outside localhost.
- Validate uploaded file size and schema before inference.
- Store artifacts in a controlled path instead of relying on `artifacts/latest`.
- Add logging and monitoring around prediction requests.
- Use a process manager or container runtime for long-running services.
