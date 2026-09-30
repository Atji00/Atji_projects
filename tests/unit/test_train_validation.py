import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

import scoring_model.eval_artefacts_func.train_validation as train_validation_module
from scoring_model.eval_artefacts_func.train_validation import LearningCurve

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def learning_curve(monkeypatch, patch_paths, eval_files, fitted_model):
    patch_paths(train_validation_module)
    monkeypatch.setattr(train_validation_module, "MODEL_NAME", "test_model")
    monkeypatch.setattr(train_validation_module, "model_trained", fitted_model)
    return LearningCurve(split=3)


# ----------------------------------------------------------------------
# 1. Learning curve computation
# ----------------------------------------------------------------------

class TestComputation:

    def test_ten_train_sizes(self, learning_curve):
        assert len(learning_curve.train_size) == 10

    def test_scores_shape_follow_number_of_folds(self, learning_curve):
        assert learning_curve.train_score.shape == (10, 3)
        assert learning_curve.val_score.shape == (10, 3)

    def test_train_sizes_are_increasing(self, learning_curve):
        sizes = list(learning_curve.train_size)
        assert sizes == sorted(sizes)

    def test_scores_are_between_zero_and_one(self, learning_curve):
        assert ((learning_curve.val_score >= 0) & (learning_curve.val_score <= 1)).all()


# ----------------------------------------------------------------------
# 2. Figure
# ----------------------------------------------------------------------

class TestFigure:

    def test_returns_figure(self, learning_curve):
        assert isinstance(learning_curve.build(), Figure)

    def test_png_is_saved(self, learning_curve, tmp_project):
        learning_curve.build()
        assert (tmp_project.figures / "test_model" / "learning_curve_test_model.png").is_file()

    def test_train_and_validation_curves_are_plotted(self, learning_curve):
        ax = learning_curve.build().axes[0]
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels == ["Train score", "Validation score"]

    def test_figure_is_closed(self, learning_curve):
        fig = learning_curve.build()
        assert not plt.fignum_exists(fig.number)
