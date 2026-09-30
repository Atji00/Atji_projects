import joblib
import pandas as pd
import pytest
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import scoring_model.features.preprocessing as preprocessing_module
from scoring_model.features.preprocessing import Processor

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def paths(patch_paths):
    return patch_paths(preprocessing_module)


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "balance": [1000.0, 2000.0, 3000.0, 4000.0],
            "campaign": [1.0, 2.0, 3.0, 4.0],
            "previous": [0.0, 1.0, 0.0, 1.0],
            "marital": ["married", "single", "divorced", "married"],
            "housing": [True, False, True, False],
            "loan": [False, False, True, True],
            "poutcome": ["unknown", "success", "failure", "other"],
            "age": [25, 30, 35, 40],
        },
        index=[10, 11, 12, 13],
    )


@pytest.fixture
def processor(paths, data):
    return Processor(data)


# ----------------------------------------------------------------------
# 1. Processor construction from training.yaml
# ----------------------------------------------------------------------

class TestBuildProcessor:

    def test_returns_column_transformer(self, processor):
        assert isinstance(processor.processor, ColumnTransformer)

    def test_scaler_and_encoder_from_config(self, processor):
        transformers = {name: transformer for name, transformer, _ in processor.processor.transformers}
        assert isinstance(transformers["numerical"], StandardScaler)
        assert isinstance(transformers["categorical"], OneHotEncoder)

    def test_encoder_parameters_from_config(self, processor):
        transformers = {name: transformer for name, transformer, _ in processor.processor.transformers}
        encoder = transformers["categorical"]
        assert encoder.handle_unknown == "ignore"
        assert encoder.drop == "first"
        assert encoder.sparse_output is False

    def test_unknown_scaler_raises(self, paths, data):
        config_path = paths.configs / "training.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["training"]["preprocessing"]["numerical"]["scaler"] = "UnknownScaler"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(AttributeError):
            Processor(data)


# ----------------------------------------------------------------------
# 2. Fit and transform
# ----------------------------------------------------------------------

class TestFitTransform:

    def test_returns_dataframe_with_same_index(self, processor, data):
        result = processor.fit_transform()
        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == list(data.index)

    def test_non_selected_columns_are_dropped(self, processor):
        result = processor.fit_transform()
        assert not any("age" in column for column in result.columns)

    def test_column_names_are_prefixed_by_transformer(self, processor):
        result = processor.fit_transform()
        assert "numerical__balance" in result.columns
        assert "categorical__marital_single" in result.columns

    def test_numerical_features_are_standardized(self, processor):
        result = processor.fit_transform()
        assert result["numerical__balance"].mean() == pytest.approx(0.0)
        assert result["numerical__balance"].std(ddof=0) == pytest.approx(1.0)

    def test_first_category_is_dropped(self, processor):
        result = processor.fit_transform()
        assert "categorical__marital_divorced" not in result.columns


# ----------------------------------------------------------------------
# 3. Saved artefacts
# ----------------------------------------------------------------------

class TestSavedArtefacts:

    def test_fitted_processor_is_saved(self, processor, paths, data):
        processor.fit_transform()
        saved = joblib.load(paths.features / "processor_fitted.joblib")
        assert saved.transform(data).shape[0] == len(data)

    def test_processed_train_is_saved(self, processor, paths):
        result = processor.fit_transform()
        saved = pd.read_parquet(paths.processed / "processed_train" / "X_train_processed.parquet")
        assert list(saved.columns) == list(result.columns)
        assert len(saved) == len(result)
