"""Descarga los precios OHLC diarios de cada ticker configurado y los guarda en data/raw.

Uso:
    python -m src.data.make_dataset                 # todos los tickers de params.yaml
    python -m src.data.make_dataset --ticker MSFT   # solo uno
"""
import argparse
import logging

import yfinance as yf

from src.config import DATA_RAW, get_tickers, load_params, raw_file

logger = logging.getLogger(__name__)


def download_prices(ticker: str, start: str, end: str):
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No se descargaron datos para {ticker}")
    df.index = df.index.tz_localize(None).normalize()
    df.index.name = "date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def main(only: list[str] | None = None):
    params = load_params()
    cfg = params["data"]
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for ticker in get_tickers(params, only):
        df = download_prices(ticker, cfg["start"], cfg["end"])
        out = raw_file(ticker)
        df.to_csv(out)
        logger.info("%s: guardadas %d filas en %s", ticker, len(df), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", action="append", help="Procesa solo este ticker (repetible)")
    main(parser.parse_args().ticker)
