import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.model_selection import StratifiedKFold, learning_curve

from scoring_model.runtime.config import ConfigLoader
from scoring_model.runtime.paths import ProjectPaths
from scoring_model.scripts.predict import MODEL_NAME, model_trained


class LearningCurve:

    def __init__(self, split: int = 10, config_file: str = "config.yaml") -> None:

        self.config = ConfigLoader().load_yaml(config_file)
        self.paths = ProjectPaths()
        self.random = self.config["rd_seed"]

    
        self.X_dir = self.paths.processed / "processed_train"
        self.y_dir = self.paths.intermediate / "unprocessed_train"

        self.X_train_processed = pd.read_parquet(self.X_dir / "X_train_processed.parquet")

        self.y_train = pd.read_parquet(self.y_dir / "y_train.parquet")

        self.model_name = MODEL_NAME
        self.model_trained = model_trained
        
        self.stratfold = StratifiedKFold(n_splits=split, shuffle=True, random_state=self.random)

        (self.train_size, 
         self.train_score, 
         self.val_score) = learning_curve(
                                            self.model_trained,
                                            self.X_train_processed,
                                            self.y_train.to_numpy().ravel(),
                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                            cv=self.stratfold,
                                            return_times=False
                                        )

    def build(self) -> Figure:

        self.output_dir = self.paths.figures / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # -------------------- Mean scores -------------------- #

        train_mean = self.train_score.mean(axis=1)
        val_mean = self.val_score.mean(axis=1)

        # -------------------- Confidence bands -------------------- #

        train_std = self.train_score.std(axis=1)
        val_std = self.val_score.std(axis=1)

        train_lower = train_mean - train_std
        train_upper = train_mean + train_std

        val_lower = val_mean - val_std
        val_upper = val_mean + val_std

    
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(self.train_size, train_mean, label="Train score")
        ax.plot(self.train_size, val_mean, label="Validation score")

        ax.fill_between(self.train_size, train_lower, train_upper, alpha=0.2)
        ax.fill_between(self.train_size, val_lower, val_upper, alpha=0.2)

        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Score")

        ax.set_title(f"Learning Curve - {self.model_name}")

        ax.legend()

        fig.tight_layout()

        fig.savefig(    
                        self.output_dir / f"learning_curve_{self.model_name}.png",
                        dpi=300,
                        bbox_inches="tight"
                    )

        return fig