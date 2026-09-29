"""ARIMA: selección de orden por AIC en train y pronóstico walk-forward en test.

En test se pronostica un paso adelante: para predecir el día t se usan los
parámetros estimados en train y los valores observados hasta t-1, igual que
las redes neuronales con su ventana de lookback.

Uso:
    python -m src.models.train_arima                 # todos los tickers de params.yaml
    python -m src.models.train_arima --ticker MSFT   # solo uno
"""
import argparse
import itertools
import logging
import warnings

import mlflow
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.config import get_tickers, load_params, models_dir, reports_dir
from src.models.common import (
    load_series,
    registered_model_name,
    regression_metrics,
    setup_mlflow,
    to_model_scale,
    to_original_scale,
    train_test_split_series,
)

logger = logging.getLogger(__name__)
MODEL_NAME = "arima"


def select_order(y_train: np.ndarray, grid: dict):
    best_aic, best_order, best_res = np.inf, None, None
    for order in itertools.product(grid["p"], grid["d"], grid["q"]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(y_train, order=order).fit()
        except Exception as exc:  # órdenes que no convergen se descartan
            logger.warning("ARIMA%s falló: %s", order, exc)
            continue
        logger.info("ARIMA%s AIC=%.2f", order, res.aic)
        if res.aic < best_aic:
            best_aic, best_order, best_res = res.aic, order, res
    return best_order, best_res


def train_ticker(ticker: str, params: dict):
    log_t = params["features"]["log_transform"]
    series = load_series(ticker)
    train, test = train_test_split_series(series, params["split"]["test_size"])

    y_train = to_model_scale(train.values, log_t)
    y_full = to_model_scale(series.values, log_t)

    with mlflow.start_run(run_name=f"{ticker}_{MODEL_NAME}_final"):
        mlflow.set_tags({"ticker": ticker, "model_type": MODEL_NAME, "run_type": "final"})
        order, res = select_order(y_train, params["arima"])
        logger.info("%s mejor orden: ARIMA%s", ticker, order)

        # Filtra la serie completa con los parámetros fijos de train (sin re-estimar)
        # y toma las predicciones un paso adelante sobre el tramo de test.
        res_full = res.apply(y_full, refit=False)
        pred = res_full.predict(start=len(train), end=len(series) - 1)
        y_pred = to_original_scale(pred, log_t)

        metrics = regression_metrics(test.values, y_pred)
        mlflow.log_params({"p": order[0], "d": order[1], "q": order[2], "aic": round(res.aic, 2)})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        preds = pd.DataFrame({"y_true": test.values, "y_pred": y_pred}, index=test.index)
        rep_dir = reports_dir(ticker)
        rep_dir.mkdir(parents=True, exist_ok=True)
        out = rep_dir / f"predictions_{MODEL_NAME}.csv"
        preds.to_csv(out)
        mlflow.log_artifact(str(out))

        mod_dir = models_dir(ticker)
        mod_dir.mkdir(parents=True, exist_ok=True)
        res.save(mod_dir / f"{MODEL_NAME}.pkl")
        mlflow.statsmodels.log_model(
            res, name="model", registered_model_name=registered_model_name(ticker, MODEL_NAME)
        )
        logger.info("%s ARIMA%s test: %s", ticker, order, metrics)


def main(only: list[str] | None = None):
    params = load_params()
    setup_mlflow(params)
    for ticker in get_tickers(params, only):
        train_ticker(ticker, params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", action="append", help="Entrena solo este ticker (repetible)")
    main(parser.parse_args().ticker)
