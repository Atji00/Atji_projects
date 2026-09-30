import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

import scoring_model.eval_artefacts_func.confusion_matrix as confusion_matrix_module
from scoring_model.eval_artefacts_func.confusion_matrix import ConfusionMatrix

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def matrix(monkeypatch, patch_paths, eval_files):
    patch_paths(confusion_matrix_module)
    monkeypatch.setattr(confusion_matrix_module, "MODEL_NAME", "test_model")
    return ConfusionMatrix()


# ----------------------------------------------------------------------
# Confusion matrix figure
# ----------------------------------------------------------------------

class TestConfusionMatrix:

    def test_returns_figure(self, matrix):
        assert isinstance(matrix.build(), Figure)

    def test_png_is_saved(self, matrix, tmp_project):
        matrix.build()
        assert (tmp_project.figures / "test_model" / "confusion_matrix_test_model.png").is_file()

    def test_title_contains_model_name(self, matrix):
        fig = matrix.build()
        assert "test_model" in fig.axes[0].get_title()

    def test_cells_sum_to_number_of_observations(self, matrix, eval_files):
        fig = matrix.build()
        values = [int(text.get_text()) for text in fig.axes[0].texts if text.get_text().isdigit()]
        assert sum(values) == len(eval_files["y_test"])

    def test_figure_is_closed(self, matrix):
        fig = matrix.build()
        assert not plt.fignum_exists(fig.number)
