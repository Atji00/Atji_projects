import joblib
import pandas as pd
import pandera.errors as pa_errors
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from scoring_model.features.selected_features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from scoring_model.scripts.train import train_orchestration

from .conftest import MODEL, MODEL_NAME, RAW_FILENAME, REAL_PATHS, VERSION

pytestmark = pytest.mark.integration

SCHEMA_ERRORS = (pa_errors.SchemaError, pa_errors.SchemaErrors)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_data():
    return pd.read_csv(REAL_PATHS.raw / RAW_FILENAME, sep=";")


@pytest.fixture(scope="module")
def split_data(trained_project):

    train_dir = trained_project.intermediate / "unprocessed_train"
    test_dir = trained_project.intermediate / "unprocessed_test"

    return {
            "X_train": pd.read_parquet(train_dir / "X_train_unprocessed.parquet"),
            "y_train": pd.read_parquet(train_dir / "y_train.parquet"),
            "X_test": pd.read_parquet(test_dir / "X_test_unprocessed.parquet"),
            "y_test": pd.read_parquet(test_dir / "y_test.parquet"),
            }


@pytest.fixture(scope="module")
def X_train_processed(trained_project):
    return pd.read_parquet(trained_project.processed / "processed_train" / "X_train_processed.parquet")


# ----------------------------------------------------------------------
# 1. Artefacts written by the training pipeline
# ----------------------------------------------------------------------

class TestArtefacts:

    @pytest.mark.parametrize("relative_path", [
        "data/intermediate/unprocessed_train/X_train_unprocessed.parquet",
        "data/intermediate/unprocessed_train/y_train.parquet",
        "data/intermediate/unprocessed_test/X_test_unprocessed.parquet",
        "data/intermediate/unprocessed_test/y_test.parquet",
        "data/processed/processed_train/X_train_processed.parquet",
        "src/scoring_model/features/processor_fitted.joblib",
        f"src/scoring_model/models/registry/{MODEL_NAME}.joblib",
    ])
    def test_artefact_is_written(self, trained_project, relative_path):
        assert (trained_project.root / relative_path).is_file()


# ----------------------------------------------------------------------
# 2. Data flow: raw -> casting -> engineering -> split
# ----------------------------------------------------------------------

class TestDataFlow:

    def test_no_row_is_lost_by_the_split(self, split_data, raw_data):
        assert len(split_data["X_train"]) + len(split_data["X_test"]) == len(raw_data)

    def test_features_and_target_are_aligned(self, split_data):
        assert len(split_data["X_train"]) == len(split_data["y_train"])
        assert len(split_data["X_test"]) == len(split_data["y_test"])

    def test_test_size_from_training_yaml(self, split_data, raw_data):
        assert len(split_data["X_test"]) == pytest.approx(0.20 * len(raw_data), abs=1)

    def test_split_is_stratified(self, split_data):
        train_rate = split_data["y_train"]["y"].mean()
        test_rate = split_data["y_test"]["y"].mean()
        assert train_rate == pytest.approx(test_rate, abs=0.01)

    def test_target_is_casted_to_boolean(self, split_data):
        assert pd.api.types.is_bool_dtype(split_data["y_train"]["y"])

    def test_target_is_removed_from_features(self, split_data):
        assert "y" not in split_data["X_train"].columns
        assert "y" not in split_data["X_test"].columns

    @pytest.mark.parametrize("column", ["total_contacts", "balance_per_contact"])
    def test_engineered_features_are_present(self, split_data, column):
        assert column in split_data["X_train"].columns
        assert column in split_data["X_test"].columns

    def test_engineered_features_are_computed_from_raw_columns(self, split_data):
        X = split_data["X_train"]
        assert (X["total_contacts"] == X["campaign"] + X["previous"]).all()

    def test_train_set_has_no_missing_values(self, split_data):
        assert not split_data["X_train"].isna().any().any()


# ----------------------------------------------------------------------
# 3. Preprocessing and model
# ----------------------------------------------------------------------

class TestPreprocessingAndModel:

    def test_processor_is_a_fitted_column_transformer(self, trained_project):
        processor = joblib.load(trained_project.features / "processor_fitted.joblib")
        assert isinstance(processor, ColumnTransformer)
        assert hasattr(processor, "transformers_")

    def test_processed_train_has_expected_columns(self, trained_project, X_train_processed):
        processor = joblib.load(trained_project.features / "processor_fitted.joblib")
        assert list(X_train_processed.columns) == list(processor.get_feature_names_out())

    def test_only_selected_features_are_kept(self, X_train_processed):
        sources = {name.split("__", 1)[1] for name in X_train_processed.columns}
        numerical = {name for name in sources if name in NUMERICAL_FEATURES}
        assert numerical == set(NUMERICAL_FEATURES)
        assert all(
                    name in NUMERICAL_FEATURES
                    or any(name.startswith(f"{feature}_") for feature in CATEGORICAL_FEATURES)
                    for name in sources
                   )

    def test_processed_train_has_same_rows_as_train(self, X_train_processed, split_data):
        assert len(X_train_processed) == len(split_data["X_train"])

    def test_processed_train_has_no_missing_values(self, X_train_processed):
        assert not X_train_processed.isna().any().any()

    def test_numerical_features_are_standardized(self, X_train_processed):
        numerical = X_train_processed[[f"numerical__{name}" for name in NUMERICAL_FEATURES]]
        assert numerical.mean().abs().max() == pytest.approx(0, abs=1e-6)
        assert numerical.std(ddof=0).to_list() == pytest.approx([1.0] * len(NUMERICAL_FEATURES))

    def test_saved_model_is_fitted_on_processed_features(self, trained_project, X_train_processed):
        model = joblib.load(trained_project.models_registry / f"{MODEL_NAME}.joblib")
        assert isinstance(model, LogisticRegression)
        assert list(model.feature_names_in_) == list(X_train_processed.columns)

    def test_saved_model_predicts_on_processed_train(self, trained_project, X_train_processed):
        model = joblib.load(trained_project.models_registry / f"{MODEL_NAME}.joblib")
        assert len(model.predict(X_train_processed)) == len(X_train_processed)


# ----------------------------------------------------------------------
# 4. Every model of models.yaml can be trained by the pipeline
# ----------------------------------------------------------------------

class TestAllModels:

    @pytest.mark.parametrize("model_name, model_class", [
        ("logistic_regression", LogisticRegression),
        ("random_forest", RandomForestClassifier),
        ("xgboost", XGBClassifier),
    ])
    def test_pipeline_trains_and_saves_model(self, trained_project, model_name, model_class):
        train_orchestration(model_name, "v_integration")
        model = joblib.load(trained_project.models_registry / f"{model_name}_v_integration.joblib")
        assert isinstance(model, model_class)

    def test_pipeline_is_reproducible(self, trained_project, split_data):
        train_orchestration("logistic_regression", "v_integration")
        X_test = pd.read_parquet(
                                    trained_project.intermediate / "unprocessed_test"
                                    / "X_test_unprocessed.parquet"
                                )
        pd.testing.assert_frame_equal(X_test, split_data["X_test"])


# ----------------------------------------------------------------------
# 5. Failure scenarios: the pipeline stops before writing artefacts
# ----------------------------------------------------------------------

class TestFailures:

    def test_missing_raw_file_raises(self, empty_project):
        with pytest.raises(FileNotFoundError):
            train_orchestration(MODEL, VERSION)

    @pytest.mark.parametrize("column, value", [
        ("age", 150),
        ("job", "astronaut"),
        ("y", "maybe"),
    ])
    def test_invalid_raw_data_is_rejected(self, empty_project, raw_data, column, value):
        invalid = raw_data.copy()
        invalid.loc[0, column] = value
        invalid.to_csv(empty_project.raw / RAW_FILENAME, sep=";", index=False)

        with pytest.raises(SCHEMA_ERRORS):
            train_orchestration(MODEL, VERSION)

        assert not empty_project.intermediate.exists()
        assert not (empty_project.models_registry / f"{MODEL_NAME}.joblib").exists()

    def test_unexpected_column_is_rejected(self, empty_project, raw_data):
        invalid = raw_data.assign(unexpected=1)
        invalid.to_csv(empty_project.raw / RAW_FILENAME, sep=";", index=False)

        with pytest.raises(SCHEMA_ERRORS):
            train_orchestration(MODEL, VERSION)

    def test_unknown_model_raises(self, trained_project):
        with pytest.raises(ValueError, match="Unknown model"):
            train_orchestration("unknown_model", "v_integration")

        assert not (trained_project.models_registry / "unknown_model_v_integration.joblib").exists()
