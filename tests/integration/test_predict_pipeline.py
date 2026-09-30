import joblib
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from scoring_model.scripts.predict import predict

from .conftest import MODEL_NAME

pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def outputs(trained_project):
    """Run the predict pipeline on the test set produced by the training pipeline."""
    X_test_processed, y_predicted, y_probability = predict()
    return {"X": X_test_processed, "y_pred": y_predicted, "y_prob": y_probability}


@pytest.fixture(scope="module")
def processed_test_dir(trained_project, outputs):
    return trained_project.processed / "processed_test"


@pytest.fixture(scope="module")
def y_test(trained_project):
    return pd.read_parquet(trained_project.intermediate / "unprocessed_test" / "y_test.parquet")


@pytest.fixture(scope="module")
def model(trained_project):
    return joblib.load(trained_project.models_registry / f"{MODEL_NAME}.joblib")


# ----------------------------------------------------------------------
# 1. Artefacts written by the predict pipeline
# ----------------------------------------------------------------------

class TestArtefacts:

    @pytest.mark.parametrize("filename", [
        "X_test_processed.parquet",
        "y_predicted.parquet",
        "y_probability.parquet",
    ])
    def test_artefact_is_written(self, processed_test_dir, filename):
        assert (processed_test_dir / filename).is_file()

    def test_saved_files_match_returned_values(self, processed_test_dir, outputs):
        y_pred = pd.read_parquet(processed_test_dir / "y_predicted.parquet")["y_pred"]
        y_prob = pd.read_parquet(processed_test_dir / "y_probability.parquet")["y_prob"]
        assert y_pred.to_list() == outputs["y_pred"].to_list()
        assert y_prob.to_list() == pytest.approx(outputs["y_prob"].to_list())


# ----------------------------------------------------------------------
# 2. Consistency between training and prediction
# ----------------------------------------------------------------------

class TestTrainPredictConsistency:

    def test_one_prediction_per_test_observation(self, outputs, y_test):
        assert len(outputs["X"]) == len(y_test)
        assert len(outputs["y_pred"]) == len(y_test)
        assert len(outputs["y_prob"]) == len(y_test)

    def test_test_features_match_train_features(self, trained_project, outputs):
        X_train = pd.read_parquet(
                                    trained_project.processed / "processed_train"
                                    / "X_train_processed.parquet"
                                  )
        assert list(outputs["X"].columns) == list(X_train.columns)

    def test_test_features_match_model_features(self, outputs, model):
        assert list(outputs["X"].columns) == list(model.feature_names_in_)

    def test_processed_test_has_no_missing_values(self, outputs):
        assert not outputs["X"].isna().any().any()

    def test_predictions_are_those_of_the_saved_model(self, outputs, model):
        assert outputs["y_pred"].to_list() == list(model.predict(outputs["X"]))


# ----------------------------------------------------------------------
# 3. Predictions validity
# ----------------------------------------------------------------------

class TestPredictions:

    def test_predictions_have_target_classes(self, outputs, y_test):
        assert set(outputs["y_pred"].unique()) <= set(y_test["y"].unique())

    def test_both_classes_are_predicted(self, outputs):
        assert outputs["y_pred"].nunique() == 2

    def test_probabilities_are_between_0_and_1(self, outputs):
        assert outputs["y_prob"].between(0, 1).all()

    def test_predictions_follow_probabilities(self, outputs):
        assert ((outputs["y_prob"] >= 0.5) == outputs["y_pred"].astype(bool)).all()

    def test_model_is_better_than_random(self, outputs, y_test):
        assert roc_auc_score(y_test["y"], outputs["y_prob"]) > 0.6

    def test_predict_is_deterministic(self, outputs):
        _, y_pred, y_prob = predict()
        assert y_pred.to_list() == outputs["y_pred"].to_list()
        assert y_prob.to_list() == pytest.approx(outputs["y_prob"].to_list())


# ----------------------------------------------------------------------
# 4. Failure scenarios
# ----------------------------------------------------------------------

class TestFailures:

    def test_predict_without_training_raises(self, empty_project):
        with pytest.raises(FileNotFoundError):
            predict()
