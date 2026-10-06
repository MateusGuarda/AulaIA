"""Etapa 1 do pipeline: carga, engenharia de features e divisão treino/validação/teste."""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

TARGET = "MedHouseVal"

# Coordenadas aproximadas (lat, lon) dos grandes centros
LOS_ANGELES = (34.05, -118.24)
SAN_FRANCISCO = (37.77, -122.42)


def load_raw(data_home: str | None = "data_cache") -> pd.DataFrame:
    ds = fetch_california_housing(as_frame=True, data_home=data_home)
    return ds.frame


def _haversine_km(lat, lon, ref):
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(ref[0]), np.radians(ref[1])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]
    df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]
    df["DistLA_km"] = _haversine_km(df["Latitude"], df["Longitude"], LOS_ANGELES)
    df["DistSF_km"] = _haversine_km(df["Latitude"], df["Longitude"], SAN_FRANCISCO)
    df["DistNearestCity_km"] = df[["DistLA_km", "DistSF_km"]].min(axis=1)
    return df


def split(df: pd.DataFrame, val_size: float, test_size: float, seed: int):
    """Divide em treino, validação e teste. val_size e test_size são frações do total."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    val_rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_rel, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test
