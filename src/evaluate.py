"""Etapa 3 do pipeline: métricas e gráficos de avaliação."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, prefix: str) -> dict:
    return {
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
    }


def plot_pred_vs_true(y_true, y_pred, title: str, path: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Valor real (x100 mil USD)")
    ax.set_ylabel("Valor previsto (x100 mil USD)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_residuals(y_true, y_pred, title: str, path: str):
    res = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(res, bins=60)
    ax.axvline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Resíduo (real menos previsto)")
    ax.set_ylabel("Frequência")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def feature_importance(pipeline, feature_names) -> pd.DataFrame | None:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_)
    else:
        return None
    return (pd.DataFrame({"feature": feature_names, "importance": values})
            .sort_values("importance", ascending=False))


def plot_importance(df_imp: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1])
    ax.set_title("Importância das features")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
