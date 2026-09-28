"""Grid search + validación cruzada temporal para LSTM y Conv2D, con MLflow y TensorBoard.

Flujo:
  1. Para cada combinación del grid se entrena en cada fold de TimeSeriesSplit
     (solo sobre el tramo de train) y se promedia el MSE de validación.
     Cada combinación es un run anidado en MLflow y cada fold una carpeta en TensorBoard.
  2. La mejor combinación (menor MSE de CV) se re-entrena con todo el train y se
     evalúa una única vez en el test hold-out.
  3. El modelo final se registra en el Model Registry de MLflow.

Uso:
    python -m src.models.train_nn --model lstm
    python -m src.models.train_nn --model cnn2d
"""
import argparse
import logging
import shutil

import keras
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.config import MODELS_DIR, REPORTS_DIR, ROOT, load_params
from src.models.common import (
    load_series,
    regression_metrics,
    setup_mlflow,
    to_model_scale,
    to_original_scale,
    train_test_split_series,
)
from src.models.cv import Fold, scale_and_window, time_series_folds
from src.models.nn_models import build_model, prepare_input

logger = logging.getLogger(__name__)


def config_id(hparams: dict) -> str:
    return "_".join(f"{k}={v}" for k, v in sorted(hparams.items()))


def fit_and_evaluate(model_type: str, hparams: dict, fold: Fold, train_cfg: dict, log_t: bool, tb_dir):
    """Entrena en fold.X_train y devuelve (modelo, métricas en escala original, predicciones)."""
    keras.backend.clear_session()
    model = build_model(model_type, **hparams)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=train_cfg["patience"], restore_best_weights=True),
        keras.callbacks.TensorBoard(log_dir=str(tb_dir)),
    ]
    # validation_split toma las últimas muestras del train (antes de barajar),
    # así el early stopping nunca ve datos de validación/test.
    model.fit(
        prepare_input(model_type, fold.X_train),
        fold.y_train,
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        validation_split=train_cfg["validation_split"],
        callbacks=callbacks,
        verbose=0,
    )
    pred_scaled = model.predict(prepare_input(model_type, fold.X_val), verbose=0).ravel()
    y_pred = to_original_scale(fold.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel(), log_t)
    y_true = to_original_scale(fold.scaler.inverse_transform(fold.y_val.reshape(-1, 1)).ravel(), log_t)
    return model, regression_metrics(y_true, y_pred), y_true, y_pred


def grid_search(model_type: str, y_train: np.ndarray, params: dict, tb_root) -> dict:
    grid = ParameterGrid(params[model_type]["grid"])
    log_t = params["features"]["log_transform"]
    best = {"cv_mse": np.inf, "hparams": None}

    for hparams in grid:
        cid = config_id(hparams)
        with mlflow.start_run(run_name=cid, nested=True):
            mlflow.set_tags({"model_type": model_type, "run_type": "cv"})
            mlflow.log_params(hparams)
            fold_metrics = []
            folds = time_series_folds(y_train, hparams["lookback"], params["cv"]["n_splits"], params["cv"]["gap"])
            for k, fold in enumerate(folds):
                _, m, _, _ = fit_and_evaluate(
                    model_type, hparams, fold, params["training"], log_t, tb_root / cid / f"fold_{k}"
                )
                mlflow.log_metrics({"fold_mse": m["mse"], "fold_rmse": m["rmse"]}, step=k)
                fold_metrics.append(m)

            cv_mse = float(np.mean([m["mse"] for m in fold_metrics]))
            cv_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
            mlflow.log_metrics({"cv_mse": cv_mse, "cv_rmse": cv_rmse})
            logger.info("%s %s -> CV MSE=%.4f RMSE=%.4f", model_type, cid, cv_mse, cv_rmse)

            if cv_mse < best["cv_mse"]:
                best = {"cv_mse": cv_mse, "cv_rmse": cv_rmse, "hparams": hparams}
    return best


def main(model_type: str):
    params = load_params()
    keras.utils.set_random_seed(params["seed"])
    log_t = params["features"]["log_transform"]

    series = load_series()
    train, test = train_test_split_series(series, params["split"]["test_size"])
    y_train = to_model_scale(train.values, log_t)
    y_full = to_model_scale(series.values, log_t)

    tb_root = ROOT / params["tensorboard"]["log_dir"] / model_type
    shutil.rmtree(tb_root, ignore_errors=True)  # evita mezclar curvas de ejecuciones anteriores

    setup_mlflow(params)
    with mlflow.start_run(run_name=f"{model_type}_final"):
        mlflow.set_tags({"model_type": model_type, "run_type": "final"})
        best = grid_search(model_type, y_train, params, tb_root)
        logger.info("Mejores hiperparámetros %s: %s", model_type, best["hparams"])
        mlflow.log_params(best["hparams"])
        mlflow.log_metrics({"cv_mse": best["cv_mse"], "cv_rmse": best["cv_rmse"]})

        # Re-entrenamiento con todo el train; el escalador solo ve el train.
        train_idx = np.arange(len(train))
        test_idx = np.arange(len(train), len(series))
        final_fold = scale_and_window(y_full, train_idx, test_idx, best["hparams"]["lookback"])
        model, metrics, y_true, y_pred = fit_and_evaluate(
            model_type, best["hparams"], final_fold, params["training"], log_t, tb_root / "final"
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        logger.info("%s test: %s", model_type, metrics)

        preds = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=test.index)
        REPORTS_DIR.mkdir(exist_ok=True)
        pred_file = REPORTS_DIR / f"predictions_{model_type}.csv"
        preds.to_csv(pred_file)
        mlflow.log_artifact(str(pred_file))
        mlflow.log_artifacts(str(tb_root / "final"), artifact_path="tensorboard")

        MODELS_DIR.mkdir(exist_ok=True)
        model.save(MODELS_DIR / f"{model_type}.keras")
        mlflow.tensorflow.log_model(model, name="model", registered_model_name=f"aapl_vol_{model_type}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "cnn2d"], required=True)
    main(parser.parse_args().model)
