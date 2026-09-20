import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import auc, roc_curve

from scoring_model.runtime.paths import ProjectPaths


class RocCurve:

    def __init__(self, y_test: pd.Series, y_prob: pd.Series, model_name: str) -> None:

        self.y_test = y_test
        self.y_prob = y_prob
        self.name = model_name

        self.paths = ProjectPaths()
        self.output_dir = self.paths.figures / self.name

    def build(self) -> Figure:

        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        roc_auc = auc(fpr, tpr)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, label=f"Courbe ROC (AUC = {roc_auc:.2f})")
        ax.plot([0, 1],[0, 1], linestyle="--")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")

        ax.set_title(f"Courbe ROC - {self.name}")

        ax.legend(loc="lower right")

        fig.tight_layout()

        fig.savefig(self.output_dir / f"roc_curve_{self.name}.png")

        return fig