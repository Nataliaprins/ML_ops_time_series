"""Validación cruzada temporal sin fuga de información.

Reglas que se respetan en cada fold:
  1. Los folds son de ventana expansiva (TimeSeriesSplit): la validación siempre
     está en el futuro respecto al entrenamiento, nunca se mezcla el orden.
  2. Se deja un `gap` de días entre el final del train y el inicio de la validación.
  3. El escalador (StandardScaler) se ajusta solo con los datos de entrenamiento del fold.
  4. Las ventanas de validación usan como entrada únicamente valores anteriores
     al día que se pronostica (información disponible en ese momento).
  5. El early stopping usa el último tramo del train del fold, no la validación.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


@dataclass
class Fold:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    scaler: StandardScaler


def make_windows(series: np.ndarray, lookback: int, target_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Para cada índice t en target_idx: X = series[t-lookback:t], y = series[t]."""
    target_idx = target_idx[target_idx >= lookback]
    X = np.stack([series[t - lookback : t] for t in target_idx])
    y = series[target_idx]
    return X, y


def scale_and_window(
    series: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray, lookback: int
) -> Fold:
    """Ajusta el escalador en train_idx y construye ventanas para train y evaluación."""
    scaler = StandardScaler().fit(series[train_idx].reshape(-1, 1))
    scaled = scaler.transform(series.reshape(-1, 1)).ravel()
    X_train, y_train = make_windows(scaled, lookback, train_idx)
    X_val, y_val = make_windows(scaled, lookback, eval_idx)
    return Fold(X_train, y_train, X_val, y_val, scaler)


def time_series_folds(series: np.ndarray, lookback: int, n_splits: int, gap: int):
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    for train_idx, val_idx in splitter.split(series):
        yield scale_and_window(series, train_idx, val_idx, lookback)
