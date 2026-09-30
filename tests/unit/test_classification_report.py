import json

import pytest
from sklearn.metrics import accuracy_score

import scoring_model.eval_artefacts_func.classification_report as classification_report_module
from scoring_model.eval_artefacts_func.classification_report import ClassificationReport

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def report(monkeypatch, patch_paths, eval_files):
    patch_paths(classification_report_module)
    monkeypatch.setattr(classification_report_module, "MODEL_NAME", "test_model")
    return ClassificationReport()


# ----------------------------------------------------------------------
# 1. Report content
# ----------------------------------------------------------------------

class TestReportContent:

    def test_returns_dictionary_with_both_classes(self, report):
        result = report.build()
        assert {"False", "True", "accuracy", "macro avg", "weighted avg"} <= set(result)

    def test_accuracy_is_correct(self, report, eval_files):
        result = report.build()
        expected = accuracy_score(eval_files["y_test"], eval_files["y_pred"])
        assert result["accuracy"] == pytest.approx(expected)

    def test_support_equals_number_of_observations(self, report, eval_files):
        result = report.build()
        assert result["weighted avg"]["support"] == len(eval_files["y_test"])


# ----------------------------------------------------------------------
# 2. Saved JSON
# ----------------------------------------------------------------------

class TestSavedJson:

    def test_json_is_saved_in_model_metrics_folder(self, report, tmp_project):
        report.build()
        assert (tmp_project.metrics / "test_model" / "classification_report_test_model.json").is_file()

    def test_saved_json_equals_returned_report(self, report):
        result = report.build()
        assert json.loads(report.output.read_text(encoding="utf-8")) == result


# ----------------------------------------------------------------------
# 3. Missing inputs
# ----------------------------------------------------------------------

class TestMissingInputs:

    def test_missing_predictions_raises(self, monkeypatch, patch_paths, eval_files, tmp_project):
        patch_paths(classification_report_module)
        (tmp_project.processed / "processed_test" / "y_predicted.parquet").unlink()
        with pytest.raises(FileNotFoundError):
            ClassificationReport()
