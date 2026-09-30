import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score

import scoring_model.eval_artefacts_func.roc_curve as roc_curve_module
from scoring_model.eval_artefacts_func.roc_curve import RocCurve

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def curve(monkeypatch, patch_paths, eval_files):
    patch_paths(roc_curve_module)
    monkeypatch.setattr(roc_curve_module, "MODEL_NAME", "test_model")
    return RocCurve()


# ----------------------------------------------------------------------
# ROC curve figure
# ----------------------------------------------------------------------

class TestRocCurve:

    def test_returns_figure(self, curve):
        assert isinstance(curve.build(), Figure)

    def test_png_is_saved(self, curve, tmp_project):
        curve.build()
        assert (tmp_project.figures / "test_model" / "roc_curve_test_model.png").is_file()

    def test_legend_shows_auc(self, curve, eval_files):
        fig = curve.build()
        expected = roc_auc_score(eval_files["y_test"], eval_files["y_prob"])
        label = fig.axes[0].get_legend().get_texts()[0].get_text()
        assert f"AUC = {expected:.2f}" in label

    def test_curve_and_random_baseline_are_plotted(self, curve):
        fig = curve.build()
        assert len(fig.axes[0].get_lines()) == 2

    def test_axes_limits(self, curve):
        ax = curve.build().axes[0]
        assert ax.get_xlim() == (0.0, 1.0)
        assert ax.get_ylim() == (0.0, 1.05)

    def test_figure_is_closed(self, curve):
        fig = curve.build()
        assert not plt.fignum_exists(fig.number)
