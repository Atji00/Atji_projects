
import pandas as pd
from sklearn.model_selection import train_test_split

from scoring_model.runtime.config import ConfigLoader
from scoring_model.runtime.paths import ProjectPaths


class DataSplitter:
    """\nSplit casted data into train and test datasets.\n"""

    def __init__(
                    self,
                    training_config: str = "training.yaml",
                    data_config: str = "data.yaml",
                    config: str = "config.yaml",
                ) -> None:

        config_loader = ConfigLoader()

        self.training_config = config_loader.load_yaml(training_config)
        self.data_config = config_loader.load_yaml(data_config)
        self.config = config_loader.load_yaml(config)

        self.split_config = self.training_config["training"]["split"]
        self.target_name = self.data_config["data"]["target"]
        self.random_state = self.config["project"]["random_state"]

        self.paths = ProjectPaths()

    def split(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
       
        X = data.drop(columns=self.target_name)
        y = data[self.target_name]

        stratify = y if self.split_config["stratify"] else None

        X_train, X_test, y_train, y_test = train_test_split(
                                                                X,
                                                                y,
                                                                test_size=self.split_config["test_size"],
                                                                shuffle=self.split_config["shuffle"],
                                                                stratify=stratify,
                                                                random_state=self.random_state
                                                            )

        self._save_split(X_train, X_test, y_train, y_test)

        return X_train, X_test, y_train, y_test

    # This is a private method
    def _save_split(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series) -> None:

        """\nSave train and test datasets as Parquet files.\n"""

        train_dir = (self.paths.intermediate/ "unprocessed_train")
        test_dir = (self.paths.intermediate/ "unprocessed_test")

        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        X_train.to_parquet(train_dir / "features_train.parquet", index=False)
        X_test.to_parquet(test_dir / "features_test.parquet", index=False)

        (
            y_train.to_frame(name=self.target_name)
                   .to_parquet(train_dir / "target_train.parquet", index=False)
        )

        (
            y_test.to_frame(name=self.target_name)
                  .to_parquet(test_dir / "target_test.parquet", index=False)
        )


