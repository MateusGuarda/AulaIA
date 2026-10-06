"""Compara os runs do experimento e salva um resumo em CSV."""
import argparse
import os

import mlflow

COLS = ["tags.mlflow.runName", "tags.modelo", "params.data.feature_engineering",
        "metrics.cv_rmse_mean", "metrics.val_rmse", "metrics.test_rmse",
        "metrics.test_mae", "metrics.test_r2", "metrics.duration_train_s"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="california-housing")
    p.add_argument("--out", default="outputs/comparacao_runs.csv")
    a = p.parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    df = mlflow.search_runs(experiment_names=[a.experiment], order_by=["metrics.val_rmse ASC"])
    cols = [c for c in COLS if c in df.columns]
    df = df[cols].rename(columns=lambda c: c.split(".", 1)[1] if "." in c else c)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
