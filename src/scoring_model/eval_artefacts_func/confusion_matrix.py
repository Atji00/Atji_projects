import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from scoring_model.runtime.paths import ProjectPaths


class ConfusionMatrix:

    def __init__(self, y_test: pd.Series, y_pred: pd.Series, model_name: str) -> None:

        self.y_test = y_test
        self.y_pred = y_pred
        self.name = model_name

        self.paths = ProjectPaths()
        self.output_dir = self.paths.figures / self.name

    def build(self) -> Figure:

        confusion = confusion_matrix(self.y_test, self.y_pred, labels=[1, 0])

        matrix = ConfusionMatrixDisplay(
                                        confusion_matrix=confusion,
                                        display_labels=["Yes: 1", "No: 0"],
                                      )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        matrix.plot(cmap="Blues", ax=ax, colorbar=False)
        
        ax.text(-0.25, 0.9, 'False Negative', c='blue')
        ax.text(0.75, 0.9, 'True Negative', c='white')
        ax.text(-0.25, -0.15, 'True Positive', c='blue')
        ax.text(0.75, -0.15, 'False Positive', c='blue')

        ax.set_title(f'Confusion Matrix for: {self.name}')
        ax.set_ylabel("Model predictions")
        ax.set_xlabel("Actual Values")

        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        fig.tight_layout()

        fig.savefig(self.output_dir / f"confusion_matrix_{self.name}.png")

        return fig