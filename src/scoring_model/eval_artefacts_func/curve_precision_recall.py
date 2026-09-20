import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, precision_recall_curve

from scoring_model.runtime.paths import ProjectPaths


class PrecisionRecallCurve:
    
    def __init__(self, y_test: pd.Series, y_prob: pd.Series, model_name: str) -> None:

        self.y_test = y_test
        self.y_prob = y_prob
        self.name = model_name

        self.paths = ProjectPaths()
        self.output_dir = self.paths.figures / self.name
        
    def build(self) -> Figure:

        precision, recall, _ = precision_recall_curve(self.y_test, self.y_prob)

        avg_precision = average_precision_score(self.y_test, self.y_prob)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(recall, precision, label=f"PR Curve (AP = {avg_precision:.2f})")
        ax.set_xlabel("Rappel")
        ax.set_ylabel("Précision")
        ax.set_title(f"Courbe Précision-Rappel - {self.name}")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        fig.savefig(self.output_dir / f"precision_recall_curve_{self.name}.png")

        return fig