# Python de l'environnement virtuel
PYTHON = .venv/Scripts/python.exe

# Commandes disponibles
.PHONY: mlflow-start train evaluate test unitest inttest

# --------------------------------------------------
# Lancer MLflow-Server
# --------------------------------------------------

mlflow-start:
	$(PYTHON) -m mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db --workers 1


# --------------------------------------------------
# Machine Learning
# --------------------------------------------------

train:
	$(PYTHON) src/scoring_model/scripts/train.py

evaluate:
	$(PYTHON) src/scoring_model/scripts/eval.py

test:
	$(PYTHON) src/scoring_model/scripts/test.py

unitest:
	$(PYTHON) tests/unit/execute_unit_tests.py

inttest:
	$(PYTHON) tests/integration/execute_integration_test.py