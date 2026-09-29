.PHONY: requirements data features train_arima train_lstm train_cnn2d train compare all mlflow_ui tensorboard clean

PYTHON = .venv/bin/python

# Por defecto se procesan todos los tickers de params.yaml.
# Para uno o varios concretos: make all TICKER=MSFT  /  make train TICKER="AAPL MSFT"
TICKER ?=
TICKER_ARGS = $(foreach t,$(TICKER),--ticker $(t))

requirements:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.data.make_dataset $(TICKER_ARGS)

features: data
	$(PYTHON) -m src.features.build_features $(TICKER_ARGS)

train_arima:
	$(PYTHON) -m src.models.train_arima $(TICKER_ARGS)

train_lstm:
	$(PYTHON) -m src.models.train_nn --model lstm $(TICKER_ARGS)

train_cnn2d:
	$(PYTHON) -m src.models.train_nn --model cnn2d $(TICKER_ARGS)

train: train_arima train_lstm train_cnn2d

compare:
	$(PYTHON) -m src.visualization.visualize $(TICKER_ARGS)

all: features train compare

mlflow_ui:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

tensorboard:
	.venv/bin/tensorboard --logdir logs/tensorboard

clean:
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
