"""
Orquestrador do pipeline de treino, validação e teste com rastreamento no MLflow.

Uso:
    python -m src.train --config configs/exp01_baseline_linear.yaml
    python -m src.train --config configs/exp02_random_forest.yaml --set model.params.n_estimators=400
"""
import argparse
import logging
import os
import tempfile
import time
from contextlib import contextmanager

import mlflow
import mlflow.sklearn
import yaml
from mlflow.models import infer_signature
from sklearn.model_selection import cross_val_score

from src import data as data_mod
from src.config import apply_overrides, flatten, load_config
from src.evaluate import (feature_importance, plot_importance, plot_pred_vs_true,
                          plot_residuals, regression_metrics)
from src.models import build_model

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
log = logging.getLogger("pipeline")


def setup_logging(log_path: str):
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
                        force=True)


@contextmanager
def stage(name: str):
    """Registra início, fim e duração de cada etapa (rastro de execução)."""
    log.info(f"[etapa] {name}: início")
    t0 = time.perf_counter()
    yield
    dur = time.perf_counter() - t0
    mlflow.log_metric(f"duration_{name}_s", dur)
    log.info(f"[etapa] {name}: fim ({dur:.2f}s)")


def log_sklearn_model(model, signature, input_example):
    """Registra o modelo de forma compatível com MLflow 2.x e 3.x."""
    import inspect
    accepted = inspect.signature(mlflow.sklearn.log_model).parameters
    kwargs = {"signature": signature, "input_example": input_example}
    kwargs["name" if "name" in accepted else "artifact_path"] = "model"
    if "skops_trusted_types" in accepted:
        # Versões recentes usam skops; modelos de árvore precisam declarar este tipo
        kwargs["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]
    mlflow.sklearn.log_model(model, **kwargs)


def run(cfg: dict):
    seed = cfg.get("seed", 42)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.get("tracking_uri", DEFAULT_TRACKING_URI))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["experiment_name"])

    workdir = tempfile.mkdtemp(prefix="run_")
    setup_logging(os.path.join(workdir, "pipeline.log"))

    with mlflow.start_run(run_name=cfg["run_name"]) as active:
        log.info(f"Run ID: {active.info.run_id} | tracking: {tracking_uri}")

        # Contexto do experimento: pergunta e hipótese
        mlflow.set_tags({
            "pergunta": cfg.get("question", ""),
            "hipotese": cfg.get("hypothesis", ""),
            "modelo": cfg["model"]["type"],
            "dataset": "California Housing (scikit-learn)",
        })
        mlflow.log_params(flatten({k: v for k, v in cfg.items() if k not in ("question", "hypothesis")}))
        mlflow.log_dict(cfg, "config/config_usada.yaml")

        # Etapa 1: dados
        with stage("load_data"):
            df = data_mod.load_raw()
            if cfg["data"].get("feature_engineering", False):
                df = data_mod.add_features(df)
            X_tr, X_val, X_te, y_tr, y_val, y_te = data_mod.split(
                df, cfg["data"]["val_size"], cfg["data"]["test_size"], seed)
            mlflow.log_params({"n_features": X_tr.shape[1], "n_train": len(X_tr),
                               "n_val": len(X_val), "n_test": len(X_te)})
            mlflow.log_input(mlflow.data.from_pandas(X_tr.assign(**{data_mod.TARGET: y_tr}),
                                                     targets=data_mod.TARGET, name="train"), context="training")
            mlflow.log_input(mlflow.data.from_pandas(X_val.assign(**{data_mod.TARGET: y_val}),
                                                     targets=data_mod.TARGET, name="validation"), context="validation")
            mlflow.log_input(mlflow.data.from_pandas(X_te.assign(**{data_mod.TARGET: y_te}),
                                                     targets=data_mod.TARGET, name="test"), context="test")
            log.info(f"Divisão: treino={len(X_tr)} val={len(X_val)} teste={len(X_te)} features={X_tr.shape[1]}")

        # Etapa 2: treino (com validação cruzada opcional no conjunto de treino)
        model = build_model(cfg["model"], seed)
        folds = cfg.get("evaluation", {}).get("cv_folds", 0)
        if folds and folds > 1:
            with stage("cross_validation"):
                scores = -cross_val_score(model, X_tr, y_tr, cv=folds,
                                          scoring="neg_root_mean_squared_error", n_jobs=-1)
                mlflow.log_metrics({"cv_rmse_mean": scores.mean(), "cv_rmse_std": scores.std()})
                log.info(f"CV RMSE: {scores.mean():.4f} +/- {scores.std():.4f}")

        with stage("train"):
            model.fit(X_tr, y_tr)
            mlflow.log_metrics(regression_metrics(y_tr, model.predict(X_tr), "train"))

        # Etapa 3: validação
        with stage("validation"):
            pred_val = model.predict(X_val)
            m_val = regression_metrics(y_val, pred_val, "val")
            mlflow.log_metrics(m_val)
            log.info(f"Validação: {m_val}")

        # Etapa 4: teste (avaliação final)
        with stage("test"):
            pred_te = model.predict(X_te)
            m_te = regression_metrics(y_te, pred_te, "test")
            mlflow.log_metrics(m_te)
            log.info(f"Teste: {m_te}")

        # Etapa 5: artefatos (gráficos, importância, modelo)
        with stage("artifacts"):
            p1 = os.path.join(workdir, "pred_vs_real_teste.png")
            plot_pred_vs_true(y_te, pred_te, f"{cfg['run_name']} (teste)", p1)
            p2 = os.path.join(workdir, "residuos_teste.png")
            plot_residuals(y_te, pred_te, f"Resíduos {cfg['run_name']}", p2)
            mlflow.log_artifact(p1, "plots")
            mlflow.log_artifact(p2, "plots")

            imp = feature_importance(model, list(X_tr.columns))
            if imp is not None:
                p3 = os.path.join(workdir, "importancia_features.png")
                plot_importance(imp, p3)
                mlflow.log_artifact(p3, "plots")
                mlflow.log_table(imp, "tables/importancia_features.json")

            signature = infer_signature(X_tr, model.predict(X_tr.head(50)))
            log_sklearn_model(model, signature, X_tr.head(3))

        log.info("Pipeline concluído.")
        mlflow.log_artifact(os.path.join(workdir, "pipeline.log"), "logs")
        return active.info.run_id


def main():
    parser = argparse.ArgumentParser(description="Pipeline California Housing com MLflow")
    parser.add_argument("--config", required=True, help="Caminho do YAML do experimento")
    parser.add_argument("--set", nargs="*", default=[], help="Sobrescritas: chave.sub=valor")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    run(cfg)


if __name__ == "__main__":
    main()
