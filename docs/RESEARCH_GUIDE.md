# Research Guide

This guide explains how to use the repository as a more disciplined research reference. The project is intentionally lightweight, but the workflow should still avoid common time-series evaluation mistakes.

## Model Variants

The training CLI supports four model names:

| Model | Recurrent layer | Output head |
| --- | --- | --- |
| `GRU` | GRU | Linear regression head |
| `GRUKAN` | GRU | KAN regression head |
| `LSTM` | LSTM | Linear regression head |
| `LSTMKAN` | LSTM | KAN regression head |

In the KAN variants, the recurrent layer encodes the input window and the final hidden representation is passed to `KANLinear` instead of a standard linear layer.

## Data Split Policy

The pipeline uses chronological splits per CSV file:

```text
train -> validation -> test
```

This matters because random splits leak future temporal patterns into training. Feature and target scalers are fitted only on the training rows. Validation and test windows are transformed using those training-only scaler parameters.

## Recommended Experiment Plan

For a quick initial comparison:

```bash
python train_local.py \
  --models GRU GRUKAN LSTM LSTMKAN \
  --max-files 3 \
  --epochs 5 \
  --copy-latest
```

For a more serious baseline:

```bash
python train_local.py \
  --models GRU GRUKAN LSTM LSTMKAN \
  --epochs 20 \
  --batch-size 1024 \
  --hidden-dim 256 \
  --n-layers 2 \
  --window-size 90 \
  --copy-latest
```

For publication-oriented experiments, repeat the same command with several seeds:

```bash
python train_local.py --models GRU GRUKAN LSTM LSTMKAN --seed 1 --run-name seed-1
python train_local.py --models GRU GRUKAN LSTM LSTMKAN --seed 2 --run-name seed-2
python train_local.py --models GRU GRUKAN LSTM LSTMKAN --seed 3 --run-name seed-3
```

## Metrics

The package includes:

- MAE: mean absolute error.
- RMSE: root mean squared error.
- sMAPE: symmetric mean absolute percentage error.

Report metrics per dataset file because aggregate metrics can hide poor behavior on smaller or noisier regions.

## Baselines to Add Before Publication

The neural models should be compared against simple forecasting baselines:

- Last-value naive: predict the most recent observation.
- Seasonal naive: predict the value from the same hour on the previous day or previous week.
- Moving average: predict the recent rolling mean.
- A classical statistical model, such as ARIMA or exponential smoothing, when appropriate.

These baselines are not currently implemented in the repository, but they are important for credible research claims.

## Recommended Reporting Table

Use a table like this for each dataset:

| Dataset | Model | Seed | MAE | RMSE | sMAPE | Best epoch | Train time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `AEP_hourly.csv` | `LSTMKAN` | `42` | ... | ... | ... | ... | ... |

Then include an aggregate table with mean and standard deviation over seeds.

## Avoid These Pitfalls

- Do not fit scalers on the full dataset before splitting.
- Do not tune hyperparameters on the test split.
- Do not report only the best seed.
- Do not compare models trained with different window sizes or different data files unless that difference is intentional.
- Do not claim that KAN is better or worse based on one short run.

## Reproducibility Checklist

Before sharing results, record:

- Git commit hash.
- Dataset files used.
- Train, validation, and test split ratios.
- Window size.
- Model name and hyperparameters.
- Seed.
- PyTorch version.
- CUDA availability and GPU name.
- Number of epochs and early-stopping settings.
- Per-file metrics.
