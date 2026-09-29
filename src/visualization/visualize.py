"""Compara los modelos finales registrados en MLflow por MSE y RMSE de test.

Genera, para cada ticker:
  - reports/<TICKER>/metrics_comparison.csv
  - reports/<TICKER>/figures/metrics_comparison.png
  - reports/<TICKER>/figures/forecast_comparison.png
y un resumen con todos los activos:
  - reports/metrics_summary.csv
  - reports/figures/metrics_summary.png
"""
import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR, figures_dir, get_tickers, load_params, reports_dir
from src.models.common import setup_mlflow

logger = logging.getLogger(__name__)
MODELS = ["arima", "lstm", "cnn2d"]
# Color fijo por modelo (no por posición en el ranking), el mismo en todas las gráficas.
MODEL_COLORS = {"arima": "#2a78d6", "lstm": "#eb6834", "cnn2d": "#1baf7a"}


def latest_final_runs(params: dict, ticker: str) -> pd.DataFrame:
    runs = mlflow.search_runs(
        experiment_names=[params["mlflow"]["experiment"]],
        filter_string=f"tags.run_type = 'final' and tags.ticker = '{ticker}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        raise RuntimeError(f"No hay runs finales de {ticker} en MLflow; entrena los modelos primero.")
    runs = runs.drop_duplicates("tags.model_type")  # el más reciente por modelo
    table = runs[["tags.model_type", "metrics.test_mse", "metrics.test_rmse", "run_id"]]
    table.columns = ["model", "test_mse", "test_rmse", "run_id"]
    return table.sort_values("test_mse").reset_index(drop=True)


def plot_metrics(table: pd.DataFrame, ticker: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["test_mse", "test_rmse"]):
        ax.bar(table["model"], table[metric], color=[MODEL_COLORS[m] for m in table["model"]])
        ax.set_title(f"{ticker} — " + metric.replace("test_", "").upper() + " en test")
        for i, v in enumerate(table[metric]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(figures_dir(ticker) / "metrics_comparison.png", dpi=120)
    plt.close(fig)


def plot_forecasts(ticker: str, last_n: int = 250):
    fig, ax = plt.subplots(figsize=(12, 5))
    actual_plotted = False
    for name in MODELS:
        path = reports_dir(ticker) / f"predictions_{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True).tail(last_n)
        if not actual_plotted:
            ax.plot(df.index, df["y_true"], color="black", lw=1, label="Real")
            actual_plotted = True
        ax.plot(df.index, df["y_pred"], lw=1, color=MODEL_COLORS[name], label=name.upper())
    ax.set_title(f"Volatilidad diaria {ticker} (%) — últimos {last_n} días del test")
    ax.set_ylabel("Volatilidad (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir(ticker) / "forecast_comparison.png", dpi=120)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame):
    """RMSE de test por activo, una barra por modelo (la escala de MSE no es comparable a simple vista)."""
    tickers = list(dict.fromkeys(summary["ticker"]))
    models = [m for m in MODELS if m in set(summary["model"])]
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(tickers)), 4))
    for j, model in enumerate(models):
        rows = summary[summary["model"] == model].set_index("ticker").reindex(tickers)
        xs = [i + (j - (len(models) - 1) / 2) * width for i in range(len(tickers))]
        bars = ax.bar(xs, rows["test_rmse"], width * 0.92, color=MODEL_COLORS[model], label=model.upper())
        ax.bar_label(bars, fmt="%.3f", fontsize=8, color="#333333")
    ax.set_xticks(range(len(tickers)), tickers)
    ax.set_ylabel("RMSE en test (pp de volatilidad)")
    ax.set_title("RMSE en test por activo y modelo")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_summary.png", dpi=120)
    plt.close(fig)


def main(only: list[str] | None = None):
    params = load_params()
    setup_mlflow(params)
    tables = []
    for ticker in get_tickers(params, only):
        table = latest_final_runs(params, ticker)
        figures_dir(ticker).mkdir(parents=True, exist_ok=True)
        table.to_csv(reports_dir(ticker) / "metrics_comparison.csv", index=False)
        plot_metrics(table, ticker)
        plot_forecasts(ticker)
        tables.append(table.assign(ticker=ticker))

    summary = pd.concat(tables)[["ticker", "model", "test_mse", "test_rmse", "run_id"]]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "metrics_summary.csv", index=False)
    plot_summary(summary)
    print(summary[["ticker", "model", "test_mse", "test_rmse"]].to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", action="append", help="Compara solo este ticker (repetible)")
    main(parser.parse_args().ticker)
