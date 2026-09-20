from pathlib import Path


class EvaluationReport:

    def __init__(
        self,
        output_path: Path,
        model_name: str,
    ) -> None:

        self.output_path = output_path
        self.model_name = model_name

    def build(
        self,
        metrics: dict[str, float],
    ) -> None:

        content = f"""# Model Evaluation Report

## Model

- Model: {self.model_name}

## Classification Metrics

| Metric | Value |
|---|---:|
"""

        for metric_name, metric_value in metrics.items():
            content += (
                f"| {metric_name} | "
                f"{metric_value:.4f} |\n"
            )

        content += f"""
## Confusion Matrix

![Confusion Matrix](../figures/{self.model_name}/confusion_matrix_{self.model_name}.png)

## ROC Curve

![ROC Curve](../figures/{self.model_name}/roc_curve_{self.model_name}.png)

## Precision-Recall Curve

![Precision-Recall Curve](../figures/{self.model_name}/precision_recall_curve_{self.model_name}.png)

## Learning Curve

![Learning Curve](../figures/{self.model_name}/learning_curve_{self.model_name}.png)
"""

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_path.write_text(
            content,
            encoding="utf-8",
        )