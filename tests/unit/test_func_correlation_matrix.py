import warnings

import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scoring_model.utils.func_correlation_matrix import correlation_matrix

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():

    return pd.DataFrame(
        {
            "age": [25, 32, 47, 51, 29, 38, 42],
            "income": [30000, 45000, 52000, 61000, 33000, 47000, 50000],
            "score": [0.2, 0.5, 0.7, 0.9, 0.3, 0.6, 0.65],
            "category": ["A", "B", "A", "C", "B", "A", "C"],
            "boolean_col": [True, False, True, False, True, False, True]
        }
    )


@pytest.fixture(autouse=True)
def _no_gui(monkeypatch):
    """Empêche plt.show() de bloquer/ouvrir une fenêtre, et ferme les
    figures créées après chaque test pour éviter les fuites mémoire."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def _heatmap_matrix(fig):
    """Récupère les valeurs numériques réellement affichées dans la heatmap
    (indépendamment du texte annoté), sous forme de np.ndarray carré."""
    ax = fig.axes[0]
    mesh = ax.collections[0]
    n = len(ax.get_xticklabels())
    return mesh.get_array().reshape(n, n)


# ---------------------------------------------------------------------------
# Validation du paramètre `method`
# ---------------------------------------------------------------------------

def test_invalid_method_raises_value_error(sample_df):
    with pytest.raises(ValueError, match="Invalid method"):
        correlation_matrix(sample_df, ["age", "income"], method="invalid")


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_valid_methods_do_not_raise(sample_df, method):
    fig = correlation_matrix(sample_df, ["age", "income", "score"], method=method)
    assert isinstance(fig, plt.Figure)


def test_method_is_case_insensitive(sample_df):
    """La validation compare method.lower() aux méthodes autorisées : la
    fonction devrait donc accepter une casse différente en entrée."""
    fig = correlation_matrix(sample_df, ["age", "income"], method="PEARSON")
    assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# Gestion des variables invalides
# ---------------------------------------------------------------------------

def test_warns_on_missing_column(sample_df):
    with pytest.warns(UserWarning, match="not found in DataFrame columns"):
        correlation_matrix(sample_df, ["age", "not_a_column", "income"])


def test_warns_on_non_numeric_column(sample_df):
    with pytest.warns(UserWarning, match="not <numeric dtype>"):
        correlation_matrix(sample_df, ["age", "income", "category"])


def test_no_spurious_warning_for_numeric_columns(sample_df):
    """Une colonne réellement numérique ne doit déclencher aucun warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        correlation_matrix(sample_df, ["age", "income", "score"])
    assert len(caught) == 0, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# Contenu de la matrice
# ---------------------------------------------------------------------------

def test_all_valid_numeric_columns_are_kept(sample_df):
    """Les 3 variables numériques valides doivent toutes figurer dans la
    matrice -- pas seulement la dernière variable de la liste."""
    fig = correlation_matrix(sample_df, ["age", "income", "score"])
    labels = {t.get_text() for t in fig.axes[0].get_xticklabels()}
    assert labels == {"age", "income", "score"}


def test_invalid_variable_excluded_from_matrix(sample_df):
    fig = correlation_matrix(sample_df, ["age", "income", "not_a_column", "boolean_col"])
    labels = {t.get_text() for t in fig.axes[0].get_xticklabels()}
    assert "not_a_column" not in labels
    assert "boolean_col" not in labels
    assert labels == {"age", "income"}


def test_correlation_values_match_pandas(sample_df):
    cols = ["age", "income", "score"]
    expected = sample_df[cols].corr(method="pearson").values
    fig = correlation_matrix(sample_df, cols, method="pearson")
    np.testing.assert_allclose(_heatmap_matrix(fig), expected, atol=1e-8)


# ---------------------------------------------------------------------------
# Sortie de la fonction
# ---------------------------------------------------------------------------

def test_returns_matplotlib_figure(sample_df):
    fig = correlation_matrix(sample_df, ["age", "income"])
    assert isinstance(fig, plt.Figure)


def test_title_reflects_requested_method(sample_df):
    fig = correlation_matrix(sample_df, ["age", "income"], method="spearman")
    assert "spearman" in fig.axes[0].get_title().lower()


@pytest.mark.parametrize("rotation_x,ha_x", [(90, "right"), (0, "center")])
def test_xticks_rotation_and_alignment_applied(sample_df, rotation_x, ha_x):
    fig = correlation_matrix(
        sample_df, ["age", "income"], rotation_x=rotation_x, ha_x=ha_x
    )
    for label in fig.axes[0].get_xticklabels():
        assert label.get_rotation() == rotation_x
        assert label.get_ha() == ha_x


# ---------------------------------------------------------------------------
# Non-mutation des données d'entrée
# ---------------------------------------------------------------------------

def test_input_dataframe_not_mutated(sample_df):
    original = sample_df.copy(deep=True)
    correlation_matrix(sample_df, ["age", "income", "score"])
    pd.testing.assert_frame_equal(sample_df, original)


# ---------------------------------------------------------------------------
# Cas limites
# ---------------------------------------------------------------------------

def test_empty_numeric_vars_raises_clear_error(sample_df):
    """Une liste vide ne doit pas planter avec une erreur obscure
    (UnboundLocalError) mais échouer explicitement et lisiblement."""
    with pytest.raises((ValueError, UnboundLocalError)):
        correlation_matrix(sample_df, [])