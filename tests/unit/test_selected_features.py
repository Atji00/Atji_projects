import pytest

from scoring_model.features.selected_features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from scoring_model.runtime.config import ConfigLoader

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def validated_columns():
    return ConfigLoader().load_yaml("data.yaml")["data"]["validation"]["columns"]


# ----------------------------------------------------------------------
# Consistency of the selected features
# ----------------------------------------------------------------------

class TestSelectedFeatures:

    def test_lists_are_not_empty(self):
        assert NUMERICAL_FEATURES
        assert CATEGORICAL_FEATURES

    def test_no_duplicates(self):
        features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        assert len(features) == len(set(features))

    def test_target_is_not_selected(self):
        assert "y" not in NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    def test_numerical_features_are_numeric_in_data_yaml(self, validated_columns):
        for feature in NUMERICAL_FEATURES:
            assert validated_columns[feature]["type"].startswith(("int", "float"))

    def test_categorical_features_are_strings_in_data_yaml(self, validated_columns):
        for feature in CATEGORICAL_FEATURES:
            assert validated_columns[feature]["type"] == "string"
