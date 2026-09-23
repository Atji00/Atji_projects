import json

import pandas as pd
from sklearn.metrics import classification_report

from scoring_model.runtime.paths import ProjectPaths
from scoring_model.scripts.predict import MODEL_NAME


class ClassificationReport:

    def __init__(self) -> None:

        self.name = MODEL_NAME
        self.paths = ProjectPaths()

        self.ypred_dir = self.paths.processed / "processed_test"
        self.ytest_dir = self.paths.intermediate / "unprocessed_test"
        self.output_dir = self.paths.metrics / self.name

        self.y_test = pd.read_parquet(self.ytest_dir / "y_test.parquet")
        self.y_pred = pd.read_parquet(self.ypred_dir / "y_predicted.parquet")

    def build(self) -> dict:

        y_test = self.y_test.to_numpy().ravel()
        y_pred = self.y_pred.to_numpy().ravel()

        report = classification_report(y_test, y_pred, output_dict=True)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output = (self.output_dir / f"classification_report_{self.name}.json")

        self.output.write_text(
                                json.dumps(report, indent=4),
                                encoding="utf-8"
                              )

        return report