"""Utilidades compartidas por los tres modelos: datos, split, métricas y MLflow."""
import mlflow
import numpy as np
import pandas as pd

from src.config import PROCESSED_FILE, ROOT


def load_series() -> pd.Series:
    return pd.read_csv(PROCESSED_FILE, index_col="date", parse_dates=True)["volatility"]


def train_test_split_series(series: pd.Series, test_size: float) -> tuple[pd.Series, pd.Series]:
    """Split temporal: el test es siempre el tramo final de la serie."""
    n_test = int(len(series) * test_size)
    return series.iloc[:-n_test], series.iloc[-n_test:]


def to_model_scale(values, log_transform: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.log(values) if log_transform else values


def to_original_scale(values, log_transform: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.exp(values) if log_transform else values


def regression_metrics(y_true, y_pred) -> dict:
    mse = float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
    return {"mse": mse, "rmse": float(np.sqrt(mse))}


def setup_mlflow(params: dict) -> None:
    uri = params["mlflow"]["tracking_uri"]
    # Rutas sqlite relativas se resuelven contra la raíz del proyecto,
    # así los scripts funcionan desde cualquier directorio.
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{ROOT / uri.removeprefix('sqlite:///')}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(params["mlflow"]["experiment"])
