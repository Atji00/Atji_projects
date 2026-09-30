import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier

import scoring_model.eval_artefacts_func.model_interpretability as interpretability_module
from scoring_model.eval_artefacts_func.model_interpretability import ModelInterpreter

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def interpreter(monkeypatch, patch_paths, eval_files, fitted_model):
    patch_paths(interpretability_module)
    monkeypatch.setattr(interpretability_module, "MODEL_NAME", "logistic_regression_test")
    monkeypatch.setattr(interpretability_module, "model_trained", fitted_model)
    return ModelInterpreter()


@pytest.fixture
def output_dir(tmp_project):
    return tmp_project.reports / "logistic_regression_test"


# ----------------------------------------------------------------------
# 1. Risk class
# ----------------------------------------------------------------------

class TestRiskClass:

    @pytest.mark.parametrize("probability, expected", [
        (0.0, "Low Risk"),
        (0.049, "Low Risk"),
        (0.05, "Medium Risk"),
        (0.149, "Medium Risk"),
        (0.15, "High Risk"),
        (0.99, "High Risk"),
    ])
    def test_thresholds(self, interpreter, probability, expected):
        assert interpreter._risk_class(probability) == expected


# ----------------------------------------------------------------------
# 2. Feature names
# ----------------------------------------------------------------------

class TestCleanFeatureNames:

    def test_transformer_prefix_is_removed(self):
        names = ["numerical__balance", "categorical__housing_1.0"]
        assert ModelInterpreter._clean_feature_names(names) == ["balance", "housing_1.0"]

    def test_name_without_prefix_is_unchanged(self):
        assert ModelInterpreter._clean_feature_names(["age"]) == ["age"]

    def test_only_first_separator_is_removed(self):
        assert ModelInterpreter._clean_feature_names(["a__b__c"]) == ["b__c"]


# ----------------------------------------------------------------------
# 3. SHAP values
# ----------------------------------------------------------------------

class TestShapValues:

    def test_one_explanation_per_observation_and_feature(self, interpreter, eval_files):
        _, shap_values = interpreter._shap_values()
        assert shap_values.values.shape == eval_files["X"].shape

    def test_feature_names_are_cleaned(self, interpreter):
        _, shap_values = interpreter._shap_values()
        assert list(shap_values.feature_names) == ["balance", "campaign", "housing_1.0"]

    def test_logistic_regression_uses_linear_explainer(self, interpreter):
        explainer, _ = interpreter._shap_values()
        assert type(explainer).__name__ == "LinearExplainer"

    def test_other_models_use_tree_explainer(self, interpreter, processed_data):
        X, y = processed_data
        interpreter.model_name = "random_forest_test"
        interpreter.model_trained = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        explainer, _ = interpreter._shap_values()
        assert type(explainer).__name__ == "TreeExplainer"


# ----------------------------------------------------------------------
# 4. Summary plots
# ----------------------------------------------------------------------

class TestSummaryPlots:

    def test_both_summary_plots_are_saved(self, interpreter, output_dir):
        interpreter.plot_shap_summary()
        assert (output_dir / "shap_summary_logistic_regression_test.png").is_file()
        assert (output_dir / "shap_summary_bar_logistic_regression_test.png").is_file()

    def test_no_figure_left_open(self, interpreter):
        interpreter.plot_shap_summary()
        assert plt.get_fignums() == []


# ----------------------------------------------------------------------
# 5. Waterfall plot
# ----------------------------------------------------------------------

class TestWaterfall:

    def test_default_client_is_saved(self, interpreter, output_dir):
        interpreter.plot_waterfall()
        assert (output_dir / "waterfall_ClientX.png").is_file()

    def test_client_series_with_custom_name(self, interpreter, eval_files, output_dir):
        client = eval_files["X"].iloc[0]
        interpreter.plot_waterfall(client_information=client, client_name="Alice")
        assert (output_dir / "waterfall_Alice.png").is_file()

    def test_invalid_client_information_raises(self, interpreter):
        with pytest.raises(TypeError):
            interpreter.plot_waterfall(client_information=[1, 2, 3])

    def test_no_figure_left_open(self, interpreter):
        interpreter.plot_waterfall()
        assert plt.get_fignums() == []

    def test_previous_open_figure_is_not_mixed_with_waterfall(self, interpreter, monkeypatch):

        # Regression: an open ROC figure used to be reused by shap (plt.gcf())
        _, ax = plt.subplots()
        ax.set_ylabel("Taux de vrais positifs")

        saved_labels = []
        original_savefig = Figure.savefig

        def recording_savefig(fig, *args, **kwargs):
            saved_labels.extend(axis.get_ylabel() for axis in fig.axes)
            return original_savefig(fig, *args, **kwargs)

        monkeypatch.setattr(Figure, "savefig", recording_savefig)

        interpreter.plot_waterfall()

        assert saved_labels
        assert "Taux de vrais positifs" not in saved_labels

    def test_title_shows_probability_and_risk_class(self, interpreter, monkeypatch, fitted_model, eval_files):

        titles = []
        original_savefig = Figure.savefig

        def recording_savefig(fig, *args, **kwargs):
            titles.append(fig._suptitle.get_text())
            return original_savefig(fig, *args, **kwargs)

        monkeypatch.setattr(Figure, "savefig", recording_savefig)

        client = eval_files["X"].iloc[0]
        probability = fitted_model.predict_proba(client.to_frame().T)[0, 1]

        interpreter.plot_waterfall(client_information=client, client_name="Alice")

        assert "Client: Alice" in titles[0]
        assert f"Probability of Default: {probability:.2%}" in titles[0]
        assert interpreter._risk_class(probability) in titles[0]
