import json
import re

import pytest

import scoring_model.eval_artefacts_func.eval_report as eval_report_module
import scoring_model.eval_artefacts_func.model_interpretability as interpretability_module
from scoring_model.eval_artefacts_func.eval_report import EvaluationReport
from scoring_model.eval_artefacts_func.model_interpretability import ModelInterpreter

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def evaluation_report(monkeypatch, patch_paths, report_dict):

    paths = patch_paths(eval_report_module)
    monkeypatch.setattr(eval_report_module, "MODEL_NAME", "test_model")

    metrics_dir = paths.metrics / "test_model"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "classification_report_test_model.json").write_text(
                                                                        json.dumps(report_dict),
                                                                        encoding="utf-8"
                                                                        )

    return EvaluationReport()


@pytest.fixture
def content(evaluation_report, tmp_project):
    evaluation_report.build()
    return (tmp_project.reports / "Readme.md").read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# 1. Markdown content
# ----------------------------------------------------------------------

class TestMarkdownContent:

    def test_model_name_is_written(self, content):
        assert "- Model: test_model" in content

    def test_accuracy_is_written_with_four_decimals(self, content, report_dict):
        assert f"- Accuracy: {report_dict['accuracy']:.4f}" in content

    @pytest.mark.parametrize("row", ["False", "True", "macro avg", "weighted avg"])
    def test_metrics_rows_are_written(self, content, report_dict, row):
        metrics = report_dict[row]
        expected = (
                    f"| {row} | {metrics['precision']:.4f} | {metrics['recall']:.4f} "
                    f"| {metrics['f1-score']:.4f} | {int(metrics['support'])} |"
                    )
        assert expected in content

    @pytest.mark.parametrize("figure", [
        "confusion_matrix", "roc_curve", "precision_recall_curve", "learning_curve",
    ])
    def test_figures_are_linked(self, content, figure):
        assert f"(figures/test_model/{figure}_test_model.png)" in content


# ----------------------------------------------------------------------
# 2. SHAP interpretability section
# ----------------------------------------------------------------------

class TestShapSection:

    def test_shap_section_is_written(self, content):
        assert "## Model Interpretability (SHAP)" in content

    @pytest.mark.parametrize("image", [
        "shap_summary_test_model.png",
        "shap_summary_bar_test_model.png",
        "waterfall_ClientX.png",
    ])
    def test_shap_plots_are_linked(self, content, image):
        assert f"(figures/test_model/{image})" in content

    def test_shap_section_is_after_learning_curve(self, content):
        assert content.index("## Learning Curve") < content.index("## Model Interpretability (SHAP)")

    def test_custom_client_name_is_used(self, evaluation_report, tmp_project):
        evaluation_report.client_name = "Alice"
        evaluation_report.build()
        content = (tmp_project.reports / "Readme.md").read_text(encoding="utf-8")
        assert "### SHAP Waterfall - Alice" in content
        assert "(figures/test_model/waterfall_Alice.png)" in content

    def test_links_match_files_written_by_model_interpreter(
        self, monkeypatch, patch_paths, report_dict, fitted_model, tmp_project
    ):
        # Same model name in both modules ("logistic_regression*" selects LinearExplainer)
        name = "logistic_regression_test"

        patch_paths(eval_report_module, interpretability_module)
        monkeypatch.setattr(eval_report_module, "MODEL_NAME", name)
        monkeypatch.setattr(interpretability_module, "MODEL_NAME", name)
        monkeypatch.setattr(interpretability_module, "model_trained", fitted_model)

        metrics_dir = tmp_project.metrics / name
        metrics_dir.mkdir(parents=True)
        (metrics_dir / f"classification_report_{name}.json").write_text(
                                                                        json.dumps(report_dict),
                                                                        encoding="utf-8"
                                                                        )

        interpreter = ModelInterpreter()
        interpreter.plot_shap_summary()
        interpreter.plot_waterfall()

        EvaluationReport().build()

        content = (tmp_project.reports / "Readme.md").read_text(encoding="utf-8")
        written = {path.name for path in (tmp_project.figures / name).iterdir()}
        linked = set(re.findall(rf"\(figures/{name}/((?:shap_|waterfall_)[^)]+)\)", content))
        assert linked == written


# ----------------------------------------------------------------------
# 3. Missing inputs
# ----------------------------------------------------------------------

class TestMissingInputs:

    def test_missing_classification_report_raises(self, evaluation_report):
        evaluation_report.name = "unknown_model"
        with pytest.raises(FileNotFoundError):
            evaluation_report.build()
