"""Construye la serie univariada de volatilidad diaria a partir de precios OHLC.

Se usan estimadores de rango intradía, que dan una medida de volatilidad por día
sin ventanas móviles (evita que targets consecutivos compartan información).
La volatilidad se expresa en puntos porcentuales diarios.
"""
import logging

import numpy as np
import pandas as pd

from src.config import DATA_RAW, PROCESSED_FILE, load_params

logger = logging.getLogger(__name__)


def parkinson(df: pd.DataFrame) -> pd.Series:
    hl = np.log(df["High"] / df["Low"])
    return np.sqrt(hl**2 / (4 * np.log(2)))


def garman_klass(df: pd.DataFrame) -> pd.Series:
    hl = np.log(df["High"] / df["Low"])
    co = np.log(df["Close"] / df["Open"])
    var = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    return np.sqrt(var.clip(lower=0))


ESTIMATORS = {"parkinson": parkinson, "garman_klass": garman_klass}


def build_volatility(df: pd.DataFrame, estimator: str) -> pd.Series:
    vol = ESTIMATORS[estimator](df) * 100
    # Días sin rango (H == L) darían log(0); se descartan.
    vol = vol[vol > 0].dropna()
    vol.name = "volatility"
    return vol


def main():
    params = load_params()
    raw = pd.read_csv(DATA_RAW / f"{params['data']['ticker']}.csv", index_col="date", parse_dates=True)
    vol = build_volatility(raw, params["features"]["estimator"])
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    vol.to_csv(PROCESSED_FILE)
    logger.info("Serie de volatilidad: %d observaciones -> %s", len(vol), PROCESSED_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
