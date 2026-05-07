from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from lstm_kan.data import load_energy_dataset
from lstm_kan.models import build_model
from lstm_kan.training import build_loaders, evaluate_by_file, fit_model, save_artifact, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRU/LSTM/KAN models locally.")
    parser.add_argument("--data-dir", default="dataset", help="Directory containing the energy CSV files.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for checkpoints and metadata.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["LSTMKAN"],
        choices=["GRU", "GRUKAN", "LSTM", "LSTMKAN"],
        help="One or more model variants to train.",
    )
    parser.add_argument("--window-size", type=int, default=90)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--run-name", default=None, help="Optional name for the run directory.")
    parser.add_argument("--copy-latest", action="store_true", help="Mirror trained artifacts into artifacts/latest.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset = load_energy_dataset(
        args.data_dir,
        window_size=args.window_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_files=args.max_files,
    )
    train_loader, val_loader = build_loaders(dataset, args.batch_size)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for model_name in args.models:
        print(f"\n=== Training {model_name} ===")
        model = build_model(
            model_name,
            input_dim=len(dataset.feature_columns),
            hidden_dim=args.hidden_dim,
            output_dim=1,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
        model, history = fit_model(
            model,
            train_loader,
            val_loader,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
            target_scaler=dataset.target_scaler,
            verbose=True,
        )
        test_metrics = evaluate_by_file(
            model,
            dataset.test_x_by_file,
            dataset.test_y_by_file,
            target_scaler=dataset.target_scaler,
            device=args.device,
            batch_size=args.batch_size,
        )

        artifact_dir = run_dir / model_name
        metadata = {
            "run_name": run_name,
            "data_dir": str(Path(args.data_dir).resolve()),
            "device": args.device,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "max_files": args.max_files,
            "dataset_files": [summary.file_name for summary in dataset.file_summaries],
            "test_metrics": test_metrics,
        }
        save_artifact(
            model,
            artifact_dir,
            model_name=model_name,
            model_kwargs={
                "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
                "output_dim": 1,
            },
            feature_columns=dataset.feature_columns,
            target_column=dataset.target_column,
            window_size=args.window_size,
            feature_scaler=dataset.feature_scaler,
            target_scaler=dataset.target_scaler,
            history=history,
            extra_metadata=metadata,
        )

        if args.copy_latest:
            latest_dir = Path(args.output_dir) / "latest" / model_name
            latest_dir.mkdir(parents=True, exist_ok=True)
            for name in ["model.pt", "scalers.pkl", "metadata.json"]:
                source = artifact_dir / name
                target = latest_dir / name
                target.write_bytes(source.read_bytes())

        summary_rows.append(
            {
                "model_name": model_name,
                "artifact_dir": str(artifact_dir),
                "best_epoch": history.best_epoch,
                "best_val_loss": history.best_val_loss,
                "best_val_smape": history.val_smape[history.best_epoch - 1] if history.best_epoch else None,
                "mean_test_smape": sum(row["smape"] for row in test_metrics) / len(test_metrics) if test_metrics else None,
            }
        )

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_rows, fh, indent=2)

    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False)
    print(f"\nTraining complete. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
