"""Rutas del proyecto y carga de parámetros."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PROCESSED_FILE = DATA_PROCESSED / "volatility.csv"


def load_params(path: Path = ROOT / "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
