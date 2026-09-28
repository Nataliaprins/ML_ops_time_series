"""Arquitecturas de redes neuronales para la serie univariada de volatilidad.

Ambas reciben la misma ventana de `lookback` días; solo cambia cómo se le da forma:
  - LSTM:   (lookback, 1) -> secuencia de un canal.
  - Conv2D: (lookback // 5, 5, 1) -> imagen de semanas x días hábiles, para que
            los filtros capturen patrones dentro de la semana y entre semanas.
"""
import keras
import numpy as np
from keras import layers

DAYS_PER_WEEK = 5


def prepare_input(model_type: str, X: np.ndarray) -> np.ndarray:
    if model_type == "lstm":
        return X[..., np.newaxis]
    if model_type == "cnn2d":
        n, lookback = X.shape
        if lookback % DAYS_PER_WEEK:
            raise ValueError(f"lookback={lookback} debe ser múltiplo de {DAYS_PER_WEEK} para Conv2D")
        return X.reshape(n, lookback // DAYS_PER_WEEK, DAYS_PER_WEEK, 1)
    raise ValueError(f"Modelo desconocido: {model_type}")


def build_lstm(lookback: int, units: int, dropout: float, learning_rate: float) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(lookback, 1)),
            layers.LSTM(units),
            layers.Dropout(dropout),
            layers.Dense(1),
        ],
        name="lstm",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse", metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


def build_cnn2d(lookback: int, filters: int, kernel_size: int, learning_rate: float) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(lookback // DAYS_PER_WEEK, DAYS_PER_WEEK, 1)),
            layers.Conv2D(filters, kernel_size, padding="same", activation="relu"),
            layers.Conv2D(filters, kernel_size, padding="same", activation="relu"),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ],
        name="cnn2d",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse", metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


BUILDERS = {"lstm": build_lstm, "cnn2d": build_cnn2d}


def build_model(model_type: str, **hparams) -> keras.Model:
    return BUILDERS[model_type](**hparams)
