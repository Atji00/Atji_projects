import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import scoring_model.models.models_loader as models_loader_module
from scoring_model.models.models_loader import Model

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def paths(patch_paths):
    return patch_paths(models_loader_module)


@pytest.fixture
def model(paths):
    model = Model()
    # Smaller forests to keep the tests fast
    model.config["random_forest"]["parameters"]["n_estimators"] = 10
    model.config["xgboost"]["parameters"]["n_estimators"] = 10
    return model


# ----------------------------------------------------------------------
# 1. Parameters validation
# ----------------------------------------------------------------------

class TestParametersValidation:

    def test_unknown_model_raises(self, model, processed_data):
        X, y = processed_data
        with pytest.raises(ValueError):
            model.train("svm", "v001", X, y)

    def test_disabled_model_raises(self, model, processed_data):
        X, y = processed_data
        model.config["logistic_regression"]["enabled"] = False
        with pytest.raises(ValueError):
            model.train("logistic_regression", "v001", X, y)

    def test_model_without_enabled_key_is_disabled(self, model, processed_data):
        X, y = processed_data
        del model.config["logistic_regression"]["enabled"]
        with pytest.raises(ValueError):
            model.train("logistic_regression", "v001", X, y)


# ----------------------------------------------------------------------
# 2. Training
# ----------------------------------------------------------------------

class TestTraining:

    @pytest.mark.parametrize("model_name, model_class", [
        ("logistic_regression", LogisticRegression),
        ("random_forest", RandomForestClassifier),
        ("xgboost", XGBClassifier),
    ])
    def test_returns_fitted_model_of_expected_class(self, model, processed_data, model_name, model_class):
        X, y = processed_data
        trained = model.train(model_name, "v001", X, y)
        assert isinstance(trained, model_class)
        assert len(trained.predict(X)) == len(X)

    def test_parameters_from_models_yaml_are_applied(self, model, processed_data):
        X, y = processed_data
        trained = model.train("logistic_regression", "v001", X, y)
        assert trained.class_weight == "balanced"
        assert trained.solver == "lbfgs"

    def test_model_without_parameters_uses_defaults(self, model, processed_data):
        X, y = processed_data
        del model.config["logistic_regression"]["parameters"]
        trained = model.train("logistic_regression", "v001", X, y)
        assert trained.class_weight is None


# ----------------------------------------------------------------------
# 3. Saving in the registry
# ----------------------------------------------------------------------

class TestSaving:

    def test_model_is_saved_with_name_and_version(self, model, processed_data, paths):
        X, y = processed_data
        model.train("logistic_regression", "v999", X, y)
        assert (paths.models_registry / "logistic_regression_v999.joblib").is_file()

    def test_saved_model_gives_same_predictions(self, model, processed_data, paths):
        X, y = processed_data
        trained = model.train("logistic_regression", "v999", X, y)
        saved = joblib.load(paths.models_registry / "logistic_regression_v999.joblib")
        assert list(saved.predict(X)) == list(trained.predict(X))
