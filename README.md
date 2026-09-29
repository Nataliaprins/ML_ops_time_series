# ML_ops_time_series

Repositorio con los códigos para un proyecto MLOps con todo el flujo de trabajo, incluyendo tracking de modelos con MLflow y visualización del aprendizaje en TensorBoard.

## Caso de uso: pronóstico de volatilidad diaria de uno o varios activos

Se pronostica la volatilidad diaria de cada activo configurado en `params.yaml` (por defecto AAPL y MSFT) un día adelante y se comparan tres modelos univariados. Cada activo se modela por separado:

| Modelo  | Entrada | Selección de hiperparámetros |
|---------|---------|------------------------------|
| ARIMA   | serie completa hasta t-1 | orden (p, d, q) por AIC en train |
| LSTM    | ventana de `lookback` días `(lookback, 1)` | grid search + CV temporal |
| Conv2D  | ventana reorganizada como imagen `(semanas, 5 días, 1)` | grid search + CV temporal |

Los modelos se comparan **solo por MSE y RMSE** sobre un conjunto de test hold-out (último 20% de la serie).

### Volatilidad diaria

Se calcula con el estimador de **Garman-Klass** a partir de los precios OHLC (también se puede usar **Parkinson**; ver `params.yaml`). Es una medida por día, sin ventanas móviles, expresada en puntos porcentuales. Los modelos se entrenan sobre `log(vol)` y las métricas se reportan en la escala original.

### Activos

Los activos se configuran en `params.yaml` con símbolos de Yahoo Finance:

```yaml
data:
  tickers: [AAPL, MSFT]      # o SPY, ^GSPC, BTC-USD, EURUSD=X...
```

Todo el flujo recorre la lista y guarda las salidas de cada activo en su propia carpeta (`data/processed/<TICKER>/`, `models/<TICKER>/`, `reports/<TICKER>/`, `logs/tensorboard/<TICKER>/`), así que los activos no se pisan entre sí. Los símbolos con caracteres especiales se normalizan para las rutas (`^GSPC` -> `GSPC`, `BTC-USD` -> `BTC_USD`).

### Validación cruzada sin fuga de información

- **Test hold-out**: el último 20% nunca se usa en el grid search; se evalúa una sola vez con el mejor modelo.
- **TimeSeriesSplit** con ventana expansiva y `gap` entre train y validación: la validación siempre está en el futuro.
- **Escalador ajustado solo en el train** de cada fold (y en el train completo para el modelo final).
- **Ventanas causales**: para pronosticar el día *t* solo se usan valores hasta *t-1*.
- **Early stopping** con el último 10% del train del fold, nunca con la validación ni el test.
- **ARIMA walk-forward**: en test se usan los parámetros estimados en train, sin re-estimar.

## Estructura (Cookiecutter Data Science)

```
├── Makefile               <- Comandos: make data, make train, make compare, ...
├── params.yaml            <- Parámetros del proyecto (datos, CV, grids)
├── requirements.txt
├── data
│   ├── external
│   ├── interim
│   ├── processed          <- <TICKER>/volatility.csv: serie lista para modelar
│   └── raw                <- <TICKER>.csv: precios OHLC descargados de Yahoo Finance
├── models                 <- <TICKER>/: modelos finales serializados (.keras / .pkl)
├── notebooks
├── references
├── reports
│   ├── <TICKER>           <- Predicciones, métricas y gráficas de cada activo
│   ├── metrics_summary.csv <- Métricas de todos los activos y modelos
│   └── figures            <- metrics_summary.png: RMSE por activo y modelo
└── src
    ├── config.py          <- Rutas y carga de params.yaml
    ├── data
    │   └── make_dataset.py
    ├── features
    │   └── build_features.py
    ├── models
    │   ├── common.py      <- Split, transformaciones, métricas, MLflow
    │   ├── cv.py          <- Validación cruzada temporal sin fuga
    │   ├── nn_models.py   <- Arquitecturas LSTM y Conv2D
    │   ├── train_arima.py
    │   └── train_nn.py    <- Grid search + CV + registro en MLflow
    └── visualization
        └── visualize.py   <- Tabla y gráficas comparativas
```

Los artefactos generados (`data/*`, `models/*`, `reports/*`, `mlflow.db`, `mlruns/`, `logs/`) están en `.gitignore`: se reproducen con `make all`.

## Uso

```bash
python3 -m venv .venv
make requirements

make all            # descarga datos, features, entrena los 3 modelos y compara (todos los tickers)
make all TICKER=MSFT            # solo un activo (debe estar en params.yaml)
make train TICKER="AAPL MSFT"   # cualquier paso acepta TICKER
```

O paso a paso:

```bash
make features       # data/raw/<TICKER>.csv -> data/processed/<TICKER>/volatility.csv
make train_arima
make train_lstm
make train_cnn2d
make compare        # reports/<TICKER>/*, reports/metrics_summary.csv y reports/figures/metrics_summary.png
```

### MLflow

```bash
make mlflow_ui      # http://127.0.0.1:5000
```

- Experimento `volatility_forecast`, compartido por todos los activos. Cada run lleva los tags `ticker`, `model_type` y `run_type`; para ver un solo activo filtra con `tags.ticker = 'MSFT'`.
- Un run `<TICKER>_<modelo>_final` por activo y modelo con `test_mse` y `test_rmse`; en las redes, cada combinación del grid es un run anidado con `cv_mse` y `cv_rmse`.
- Modelos registrados en el Model Registry, uno por activo y modelo: `aapl_vol_arima`, `msft_vol_lstm`, etc.

### TensorBoard

```bash
make tensorboard    # http://localhost:6006
```

Las curvas de pérdida están organizadas como `logs/tensorboard/<TICKER>/<modelo>/<hiperparámetros>/fold_k` y `logs/tensorboard/<TICKER>/<modelo>/final`.
