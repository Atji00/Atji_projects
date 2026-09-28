# commande pour lancer le server mlflow
.PHONY: mlflow-start

mlflow-start:
	python -m mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db --workers 1