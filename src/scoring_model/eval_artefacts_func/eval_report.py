import json

from scoring_model.runtime.paths import ProjectPaths
from scoring_model.scripts.predict import MODEL_NAME


class EvaluationReport:

    def __init__(self, client_name: str = "ClientX") -> None:

        self.paths = ProjectPaths()
        self.name = MODEL_NAME

        # Client explained by ModelInterpreter.plot_waterfall
        self.client_name = client_name

        self.metrics_dir = self.paths.metrics / self.name
        self.figures_dir = self.paths.figures / self.name
        self.output_dir = self.paths.reports / "Readme.md"

    def _load_classification_report(self) -> dict:

        path = self.metrics_dir / f"classification_report_{self.name}.json"

        return json.loads(path.read_text(encoding="utf-8"))

    def build(self) -> None:

        classification_report = (self._load_classification_report())

        content = f"""# Model Evaluation Report

## Model

- Model: {self.name}

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
"""

        for classe in ["False", "True"]:

            metrics = classification_report[classe]

            content += (
                f"| {classe} "
                f"| {metrics['precision']:.4f} "
                f"| {metrics['recall']:.4f} "
                f"| {metrics['f1-score']:.4f} "
                f"| {int(metrics['support'])} |\n"
            )

        content += "\n"

        content += f"""
| Aggregate | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
"""

        for aggregate in ["macro avg", "weighted avg"]:

            metrics = classification_report[aggregate]

            content += (
                f"| {aggregate} "
                f"| {metrics['precision']:.4f} "
                f"| {metrics['recall']:.4f} "
                f"| {metrics['f1-score']:.4f} "
                f"| {int(metrics['support'])} |\n"
            )

        content += f"""

## Accuracy

- Accuracy: {classification_report['accuracy']:.4f}

## Confusion Matrix

![Confusion Matrix](figures/{self.name}/confusion_matrix_{self.name}.png)

## ROC Curve

![ROC Curve](figures/{self.name}/roc_curve_{self.name}.png)

## Precision-Recall Curve

![Precision-Recall Curve](figures/{self.name}/precision_recall_curve_{self.name}.png)

## Learning Curve

![Learning Curve](figures/{self.name}/learning_curve_{self.name}.png)

## Model Interpretability (SHAP)

### SHAP Summary

![SHAP Summary](figures/{self.name}/shap_summary_{self.name}.png)

### SHAP Feature Importance

![SHAP Feature Importance](figures/{self.name}/shap_summary_bar_{self.name}.png)

### SHAP Waterfall - {self.client_name}

![SHAP Waterfall](figures/{self.name}/waterfall_{self.client_name}.png)
"""

        self.output_dir.parent.mkdir(parents=True, exist_ok=True)

        self.output_dir.write_text(content, encoding="utf-8")