"""Rutas del proyecto y carga de parámetros."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def load_params(path: Path = ROOT / "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_tickers(params: dict, only: list[str] | None = None) -> list[str]:
    """Tickers a procesar: los de params.yaml, o el subconjunto pedido por CLI."""
    tickers = params["data"]["tickers"]
    if isinstance(tickers, str):
        tickers = [tickers]
    if only:
        unknown = set(only) - set(tickers)
        if unknown:
            raise ValueError(f"Tickers no configurados en params.yaml: {sorted(unknown)}")
        tickers = [t for t in tickers if t in only]
    return tickers


def ticker_slug(ticker: str) -> str:
    """Nombre seguro para rutas y Model Registry: '^GSPC' -> 'GSPC', 'BTC-USD' -> 'BTC_USD'."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")


# Salidas por activo: cada ticker tiene su propia carpeta para no pisarse entre sí.
def raw_file(ticker: str) -> Path:
    return DATA_RAW / f"{ticker_slug(ticker)}.csv"


def processed_file(ticker: str) -> Path:
    return DATA_PROCESSED / ticker_slug(ticker) / "volatility.csv"


def models_dir(ticker: str) -> Path:
    return MODELS_DIR / ticker_slug(ticker)


def reports_dir(ticker: str) -> Path:
    return REPORTS_DIR / ticker_slug(ticker)


def figures_dir(ticker: str) -> Path:
    return reports_dir(ticker) / "figures"
