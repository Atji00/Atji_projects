import pandas as pd
import pandera.pandas as pa
import pytest

from scoring_model.dataset.validation import DataValidator

SCHEMA_ERRORS = (pa.errors.SchemaError, pa.errors.SchemaErrors)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def validator(patch_paths):
    patch_paths()
    return DataValidator()


@pytest.fixture
def valid_data():
    return pd.DataFrame(
        {
            "age": [30, 45],
            "job": ["unemployed", "admin."],
            "marital": ["married", "single"],
            "education": ["primary", "tertiary"],
            "default": ["no", "yes"],
            "balance": [1787, -200],
            "housing": ["no", "yes"],
            "loan": ["no", "no"],
            "contact": ["cellular", "unknown"],
            "day": [19, 1],
            "month": ["oct", "may"],
            "duration": [79, 0],
            "campaign": [1, 3],
            "pdays": [-1, 10],
            "previous": [0, 2],
            "poutcome": ["unknown", "success"],
            "y": ["no", "yes"],
        }
    )


# ----------------------------------------------------------------------
# 1. Schema construction
# ----------------------------------------------------------------------

class TestSchema:

    def test_schema_contains_all_columns_and_target(self, validator, valid_data):
        assert set(validator.schema.columns) == set(valid_data.columns)

    def test_schema_is_strict(self, validator):
        assert validator.schema.strict is True

    def test_target_is_not_nullable(self, validator):
        assert validator.schema.columns["y"].nullable is False


# ----------------------------------------------------------------------
# 2. Valid data
# ----------------------------------------------------------------------

class TestValidData:

    def test_valid_data_is_returned(self, validator, valid_data):
        result = validator.validate(valid_data)
        pd.testing.assert_frame_equal(result, valid_data)

    def test_missing_values_are_allowed_on_nullable_columns(self, validator, valid_data):
        valid_data["job"] = ["admin.", None]
        validator.validate(valid_data)


# ----------------------------------------------------------------------
# 3. Invalid data
# ----------------------------------------------------------------------

class TestInvalidData:

    @pytest.mark.parametrize("column, value", [
        ("age", 17),
        ("age", 101),
        ("day", 32),
        ("duration", -1),
        ("campaign", 0),
        ("pdays", -2),
        ("previous", -1),
    ])
    def test_numeric_value_out_of_range_raises(self, validator, valid_data, column, value):
        valid_data.loc[0, column] = value
        with pytest.raises(SCHEMA_ERRORS):
            validator.validate(valid_data)

    @pytest.mark.parametrize("column", ["job", "marital", "month", "poutcome"])
    def test_category_not_allowed_raises(self, validator, valid_data, column):
        valid_data.loc[0, column] = "not_allowed"
        with pytest.raises(SCHEMA_ERRORS):
            validator.validate(valid_data)

    def test_target_value_not_allowed_raises(self, validator, valid_data):
        valid_data.loc[0, "y"] = "maybe"
        with pytest.raises(SCHEMA_ERRORS):
            validator.validate(valid_data)

    def test_missing_target_value_raises(self, validator, valid_data):
        valid_data.loc[0, "y"] = None
        with pytest.raises(SCHEMA_ERRORS):
            validator.validate(valid_data)

    def test_wrong_dtype_raises(self, validator, valid_data):
        valid_data["age"] = valid_data["age"].astype(float)
        with pytest.raises(SCHEMA_ERRORS):
            validator.validate(valid_data)

    def test_extra_column_raises(self, validator, valid_data):
        valid_data["unexpected"] = 1
        with pytest.raises(SCHEMA_ERRORS):
            validator.validate(valid_data)

    def test_missing_column_raises(self, validator, valid_data):
        with pytest.raises(SCHEMA_ERRORS):
            validator.validate(valid_data.drop(columns="age"))
