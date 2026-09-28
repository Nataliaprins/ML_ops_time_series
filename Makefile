.PHONY: requirements data features train_arima train_lstm train_cnn2d train compare all mlflow_ui tensorboard clean

PYTHON = .venv/bin/python

requirements:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.data.make_dataset

features: data
	$(PYTHON) -m src.features.build_features

train_arima:
	$(PYTHON) -m src.models.train_arima

train_lstm:
	$(PYTHON) -m src.models.train_nn --model lstm

train_cnn2d:
	$(PYTHON) -m src.models.train_nn --model cnn2d

train: train_arima train_lstm train_cnn2d

compare:
	$(PYTHON) -m src.visualization.visualize

all: features train compare

mlflow_ui:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

tensorboard:
	.venv/bin/tensorboard --logdir logs/tensorboard

clean:
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
