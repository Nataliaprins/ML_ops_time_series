"""Descarga los precios OHLC diarios del ticker configurado y los guarda en data/raw."""
import logging

import yfinance as yf

from src.config import DATA_RAW, load_params

logger = logging.getLogger(__name__)


def download_prices(ticker: str, start: str, end: str):
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No se descargaron datos para {ticker}")
    df.index = df.index.tz_localize(None).normalize()
    df.index.name = "date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def main():
    params = load_params()["data"]
    df = download_prices(params["ticker"], params["start"], params["end"])
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW / f"{params['ticker']}.csv"
    df.to_csv(out)
    logger.info("Guardadas %d filas en %s", len(df), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
