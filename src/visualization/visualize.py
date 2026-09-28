"""Compara los modelos finales registrados en MLflow por MSE y RMSE de test.

Genera:
  - reports/metrics_comparison.csv
  - reports/figures/metrics_comparison.png
  - reports/figures/forecast_comparison.png
"""
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR, load_params
from src.models.common import setup_mlflow

logger = logging.getLogger(__name__)
MODELS = ["arima", "lstm", "cnn2d"]


def latest_final_runs(params: dict) -> pd.DataFrame:
    runs = mlflow.search_runs(
        experiment_names=[params["mlflow"]["experiment"]],
        filter_string="tags.run_type = 'final' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        raise RuntimeError("No hay runs finales en MLflow; entrena los modelos primero.")
    runs = runs.drop_duplicates("tags.model_type")  # el más reciente por modelo
    table = runs[["tags.model_type", "metrics.test_mse", "metrics.test_rmse", "run_id"]]
    table.columns = ["model", "test_mse", "test_rmse", "run_id"]
    return table.sort_values("test_mse").reset_index(drop=True)


def plot_metrics(table: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["test_mse", "test_rmse"]):
        ax.bar(table["model"], table[metric], color="#4C72B0")
        ax.set_title(metric.replace("test_", "").upper() + " en test")
        for i, v in enumerate(table[metric]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=120)
    plt.close(fig)


def plot_forecasts(last_n: int = 250):
    fig, ax = plt.subplots(figsize=(12, 5))
    actual_plotted = False
    for name in MODELS:
        path = REPORTS_DIR / f"predictions_{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True).tail(last_n)
        if not actual_plotted:
            ax.plot(df.index, df["y_true"], color="black", lw=1, label="Real")
            actual_plotted = True
        ax.plot(df.index, df["y_pred"], lw=1, label=name.upper())
    ax.set_title(f"Volatilidad diaria AAPL (%) — últimos {last_n} días del test")
    ax.set_ylabel("Volatilidad (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_comparison.png", dpi=120)
    plt.close(fig)


def main():
    params = load_params()
    setup_mlflow(params)
    table = latest_final_runs(params)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "metrics_comparison.csv", index=False)
    plot_metrics(table)
    plot_forecasts()
    print(table[["model", "test_mse", "test_rmse"]].to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
