# ML_ops_time_series

Repositorio con los códigos para un proyecto MLOps con todo el flujo de trabajo, incluyendo tracking de modelos con MLflow y visualización del aprendizaje en TensorBoard.

## Caso de uso: pronóstico de volatilidad diaria de AAPL

Se pronostica la volatilidad diaria de la acción de Apple (AAPL) un día adelante y se comparan tres modelos univariados:

| Modelo  | Entrada | Selección de hiperparámetros |
|---------|---------|------------------------------|
| ARIMA   | serie completa hasta t-1 | orden (p, d, q) por AIC en train |
| LSTM    | ventana de `lookback` días `(lookback, 1)` | grid search + CV temporal |
| Conv2D  | ventana reorganizada como imagen `(semanas, 5 días, 1)` | grid search + CV temporal |

Los modelos se comparan **solo por MSE y RMSE** sobre un conjunto de test hold-out (último 20% de la serie).

### Volatilidad diaria

Se calcula con el estimador de **Garman-Klass** a partir de los precios OHLC (también se puede usar **Parkinson**; ver `params.yaml`). Es una medida por día, sin ventanas móviles, expresada en puntos porcentuales. Los modelos se entrenan sobre `log(vol)` y las métricas se reportan en la escala original.

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
│   ├── processed          <- Serie de volatilidad lista para modelar
│   └── raw                <- Precios OHLC descargados de Yahoo Finance
├── models                 <- Modelos finales serializados (.keras / .pkl)
├── notebooks
├── references
├── reports
│   └── figures            <- Gráficas de comparación
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

make all            # descarga datos, features, entrena los 3 modelos y compara
```

O paso a paso:

```bash
make features       # data/raw/AAPL.csv -> data/processed/volatility.csv
make train_arima
make train_lstm
make train_cnn2d
make compare        # reports/metrics_comparison.csv y reports/figures/*.png
```

### MLflow

```bash
make mlflow_ui      # http://127.0.0.1:5000
```

- Experimento `aapl_volatility`.
- Un run `*_final` por modelo con `test_mse` y `test_rmse`; en las redes, cada combinación del grid es un run anidado con `cv_mse` y `cv_rmse`.
- Modelos registrados en el Model Registry: `aapl_vol_arima`, `aapl_vol_lstm`, `aapl_vol_cnn2d`.

### TensorBoard

```bash
make tensorboard    # http://localhost:6006
```

Las curvas de pérdida están organizadas como `logs/tensorboard/<modelo>/<hiperparámetros>/fold_k` y `logs/tensorboard/<modelo>/final`.
