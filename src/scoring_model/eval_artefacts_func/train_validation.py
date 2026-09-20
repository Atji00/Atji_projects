import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from scoring_model.runtime.paths import ProjectPaths


class LearningCurve:

    def __init__(self, train_size: pd.Series, train_score: pd.Series,
                       val_score: pd.Series, model_name: str) -> None:

        self.train_size = train_size
        self.train_score = train_score
        self.val_score = val_score
        self.name = model_name

        self.paths = ProjectPaths()
        self.output_dir = self.paths.figures / self.name

    def build(self) -> Figure:

        self.output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(self.train_size, self.train_score, label="Train score")
        ax.plot(self.train_size, self.val_score, label="Validation score")

        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Training Set Size")

        ax.set_title(f"Train vs Validation Score - {self.name}")
        ax.legend()

        fig.tight_layout()

        fig.savefig(self.output_dir / f"learning_curve_{self.name}.png")

        return fig