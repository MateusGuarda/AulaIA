"""Etapa 2 do pipeline: construção do modelo a partir da configuração."""
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REGISTRY = {
    "linear_regression": (LinearRegression, True),
    "ridge": (Ridge, True),
    "random_forest": (RandomForestRegressor, False),
    "hist_gradient_boosting": (HistGradientBoostingRegressor, False),
}


def build_model(model_cfg: dict, seed: int) -> Pipeline:
    mtype = model_cfg["type"]
    if mtype not in REGISTRY:
        raise ValueError(f"Modelo desconhecido: {mtype}. Opções: {list(REGISTRY)}")
    cls, needs_scaling = REGISTRY[mtype]
    params = dict(model_cfg.get("params") or {})
    if "random_state" in cls().get_params():
        params.setdefault("random_state", seed)
    steps = []
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", cls(**params)))
    return Pipeline(steps)
