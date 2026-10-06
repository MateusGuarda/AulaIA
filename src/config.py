"""Leitura de configuração YAML com suporte a sobrescrita via linha de comando."""
import copy
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Aplica sobrescritas no formato chave.sub=valor (ex.: model.params.n_estimators=300)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def flatten(d: dict, parent: str = "") -> dict:
    """Achata o dicionário para registrar como parâmetros no MLflow."""
    out = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out
