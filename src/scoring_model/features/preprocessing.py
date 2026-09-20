import joblib
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer

from scoring_model.runtime.config import ConfigLoader
from scoring_model.runtime.paths import ProjectPaths

from .selected_features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


class Processor:
    """
    Preprocess selected numerical and categorical features.
    """

    def __init__(self, data: pd.DataFrame, config_filename: str = "training.yaml") -> None:

        self.data = data

        self.config = ConfigLoader().load_yaml(config_filename)

        self.paths = ProjectPaths()

        self.preprocessing_config = (self.config["training"]["preprocessing"])

        self.numerical_features = NUMERICAL_FEATURES

        self.categorical_features = CATEGORICAL_FEATURES

        self.processor = self._build_processor()

    def _build_processor(self) -> ColumnTransformer:
        """
        Build the ColumnTransformer from training.yaml.
        """
        #-------------------------------------------------------------
        numerical_config = self.preprocessing_config["numerical"]

        scaler_name = numerical_config["scaler"]

        scaler_parameters = numerical_config.get("parameters",{})

        scaler_class = getattr(preprocessing, scaler_name)

        scaler = scaler_class(**scaler_parameters)

        #--------------------------------------------------------------
        categorical_config = (self.preprocessing_config["categorical"])

        encoder_name = categorical_config["encoder"]

        encoder_parameters = categorical_config.get("parameters",{})

        encoder_class = getattr(preprocessing, encoder_name)

        encoder = encoder_class(**encoder_parameters)

        return ColumnTransformer(
                                    transformers=[
                                                    (
                                                        "numerical",
                                                        scaler,
                                                        self.numerical_features,
                                                    ),
                                                    (
                                                        "categorical",
                                                        encoder,
                                                        self.categorical_features,
                                                    ),
                                                ],
                                    remainder="drop"
                                )

    def fit_transform(self) -> pd.DataFrame:
        """
        Fit the processor, save the fitted processor,
        transform the data and return the transformed DataFrame.
        """

        transformed_data = self.processor.fit_transform(self.data)

        processor_path = (self.paths.features / "processor_fitted.joblib")

        feature_names = (self.processor.get_feature_names_out())

        data_processed = pd.DataFrame(
                                        transformed_data,
                                        columns=feature_names,
                                        index=self.data.index,
                                    )

        joblib.dump(self.processor, processor_path)

        output_dir = self.paths.processed / "processed_train"
        output_dir.mkdir(parents=True, exist_ok=True)

        data_processed.to_parquet(output_dir / "X_train_processed.parquet", index=False)

        return data_processed