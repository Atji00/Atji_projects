import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from scoring_model.runtime.config import ConfigLoader
from scoring_model.runtime.paths import ProjectPaths


class Model:

    def __init__(self, config_filename: str = "models.yaml") -> None:

        self.config = ConfigLoader().load_yaml(config_filename)
        self.paths = ProjectPaths()

        self.model_classes = {
                                "logistic_regression": LogisticRegression,
                                "random_forest": RandomForestClassifier,
                                "xgboost": XGBClassifier,
                            }

    def train(self, model_name: str, version: str, 
                    X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:

        # Vérification du modèle demandé
        if model_name not in self.model_classes:

            raise ValueError(
                                f"Unknown model '{model_name}'. "
                                f"Available models: {list(self.model_classes)}"
                            )

        # Récupération de la configuration
        model_config = self.config[model_name]

        if not model_config.get("enabled", False):

            raise ValueError(
                                f"Model '{model_name}' is disabled in models.yaml."
                            )

        parameters = model_config.get("parameters", {})

        # Instanciation du modèle
        model_class = self.model_classes[model_name]
        model = model_class(**parameters)

        # Entraînement
        model.fit(X_train, y_train)

        # Nom du fichier
        model_filename = f"{model_name}_{version}.joblib"

        # Dossier de sortie
        output_path = self.paths.models_registry / model_filename

        # Sauvegarde
        joblib.dump(model, output_path)

        return model
