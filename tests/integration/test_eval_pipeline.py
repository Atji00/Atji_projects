import json
import re

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from scoring_model.eval_artefacts_func.classification_report import ClassificationReport
from scoring_model.scripts.eval import eval_orchestration

from .conftest import MODEL_NAME

pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def report(evaluated_project):
    path = evaluated_project.metrics / MODEL_NAME / f"classification_report_{MODEL_NAME}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme(evaluated_project):
    return (evaluated_project.reports / "Readme.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def y_test(evaluated_project):
    return pd.read_parquet(evaluated_project.intermediate / "unprocessed_test" / "y_test.parquet")["y"]


@pytest.fixture(scope="module")
def y_pred(evaluated_project):
    return pd.read_parquet(evaluated_project.processed / "processed_test" / "y_predicted.parquet")["y_pred"]


# ----------------------------------------------------------------------
# 1. Artefacts written by the evaluation pipeline
# ----------------------------------------------------------------------

class TestArtefacts:

    @pytest.mark.parametrize("relative_path", [
        f"metrics/{MODEL_NAME}/classification_report_{MODEL_NAME}.json",
        f"figures/{MODEL_NAME}/confusion_matrix_{MODEL_NAME}.png",
        f"figures/{MODEL_NAME}/roc_curve_{MODEL_NAME}.png",
        f"figures/{MODEL_NAME}/precision_recall_curve_{MODEL_NAME}.png",
        f"figures/{MODEL_NAME}/learning_curve_{MODEL_NAME}.png",
        f"figures/{MODEL_NAME}/shap_summary_{MODEL_NAME}.png",
        f"figures/{MODEL_NAME}/shap_summary_bar_{MODEL_NAME}.png",
        f"figures/{MODEL_NAME}/waterfall_ClientX.png",
        "Readme.md",
    ])
    def test_artefact_is_written(self, evaluated_project, relative_path):
        path = evaluated_project.reports / relative_path
        assert path.is_file()
        assert path.stat().st_size > 0

    def test_figures_are_valid_png(self, evaluated_project):
        figures = list(evaluated_project.reports.rglob("*.png"))
        assert figures
        for figure in figures:
            assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ----------------------------------------------------------------------
# 2. Metrics consistency with the predict pipeline outputs
# ----------------------------------------------------------------------

class TestMetrics:

    def test_report_has_both_classes(self, report):
        assert {"False", "True", "accuracy", "macro avg", "weighted avg"} <= set(report)

    def test_support_equals_test_size(self, report, y_test):
        assert report["False"]["support"] + report["True"]["support"] == len(y_test)

    def test_support_matches_class_distribution(self, report, y_test):
        assert report["True"]["support"] == y_test.sum()

    def test_accuracy_matches_saved_predictions(self, report, y_test, y_pred):
        assert report["accuracy"] == pytest.approx(accuracy_score(y_test, y_pred))

    def test_rebuilding_report_gives_same_metrics(self, report):
        assert ClassificationReport().build() == report


# ----------------------------------------------------------------------
# 3. Markdown report
# ----------------------------------------------------------------------

class TestReadme:

    def test_model_name_is_written(self, readme):
        assert f"- Model: {MODEL_NAME}" in readme

    def test_accuracy_is_written(self, readme, report):
        assert f"- Accuracy: {report['accuracy']:.4f}" in readme

    def test_every_linked_image_exists(self, readme, evaluated_project):
        links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
        assert len(links) == 7
        missing = [link for link in links if not (evaluated_project.reports / link).is_file()]
        assert missing == []


# ----------------------------------------------------------------------
# 4. Failure scenarios
# ----------------------------------------------------------------------

class TestFailures:

    def test_eval_without_predictions_raises(self, trained_project, empty_project):
        with pytest.raises(FileNotFoundError):
            eval_orchestration()
