import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score

import scoring_model.eval_artefacts_func.curve_precision_recall as precision_recall_module
from scoring_model.eval_artefacts_func.curve_precision_recall import (
    PrecisionRecallCurve,
)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def curve(monkeypatch, patch_paths, eval_files):
    patch_paths(precision_recall_module)
    monkeypatch.setattr(precision_recall_module, "MODEL_NAME", "test_model")
    return PrecisionRecallCurve()


# ----------------------------------------------------------------------
# Precision-Recall curve figure
# ----------------------------------------------------------------------

class TestPrecisionRecallCurve:

    def test_returns_figure(self, curve):
        assert isinstance(curve.build(), Figure)

    def test_png_is_saved(self, curve, tmp_project):
        curve.build()
        assert (tmp_project.figures / "test_model" / "precision_recall_curve_test_model.png").is_file()

    def test_legend_shows_average_precision(self, curve, eval_files):
        fig = curve.build()
        expected = average_precision_score(eval_files["y_test"], eval_files["y_prob"])
        label = fig.axes[0].get_legend().get_texts()[0].get_text()
        assert f"AP = {expected:.2f}" in label

    def test_axis_labels(self, curve):
        ax = curve.build().axes[0]
        assert ax.get_xlabel() == "Rappel"
        assert ax.get_ylabel() == "Précision"

    def test_figure_is_closed(self, curve):
        fig = curve.build()
        assert not plt.fignum_exists(fig.number)
