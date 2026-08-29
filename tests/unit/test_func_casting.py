import numpy as np
import pandas as pd
import pytest

from scoring_model.utils.func_casting import casting

# ----------------------------------------------------------------------
# Empty series
# ----------------------------------------------------------------------

class TestEmptyOrAllMissing:

    def test_all_values_missing_object_series_unchanged(self):
        serie = pd.Series([np.nan, None, np.nan], name="col", dtype="object")
        result = casting(serie)
        pd.testing.assert_series_equal(result, serie)

    def test_completely_empty_series_unchanged(self):
        serie = pd.Series([], dtype="object", name="empty_col")
        result = casting(serie)
        pd.testing.assert_series_equal(result, serie)


# ----------------------------------------------------------------------
# 2. Columns already typed
# ----------------------------------------------------------------------

class TestAlreadyTypedColumns:

    def test_int64_is_cast_to_nullable_Int64(self):
        serie = pd.Series([1, 2, 3], name="ints", dtype="int64")
        result = casting(serie)
        assert result.dtype == "Int64"
        assert list(result) == [1, 2, 3]

    def test_float32_is_cast_to_float64(self):
        serie = pd.Series([1.5, 2.5, 3.5], name="floats", dtype="float32")
        result = casting(serie)
        assert result.dtype == "float64"

    def test_bool_dtype_returned_unchanged(self):
        serie = pd.Series([True, False, True], name="flags")
        result = casting(serie)
        pd.testing.assert_series_equal(result, serie)

    def test_datetime_dtype_returned_unchanged(self):
        serie = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-02"]), name="dates")
        result = casting(serie)
        pd.testing.assert_series_equal(result, serie)

    def test_categorical_dtype_returned_unchanged(self):
        serie = pd.Series(["a", "b", "a"], name="cat", dtype="object").astype("category")
        result = casting(serie)
        pd.testing.assert_series_equal(result, serie)


# ----------------------------------------------------------------------
# 3. Detect boolean columns
# ----------------------------------------------------------------------

class TestBooleanDetection:

    @pytest.mark.parametrize("values", [
        ["true", "false", "true"],
        ["Vrai", "Faux", "vrai"],
        ["oui", "non", "oui"],
        ["Yes", "No", "yes"],
        [" true ", "FALSE", "True"]
    ])
    def test_various_boolean_spellings_are_detected(self, values):
        serie = pd.Series(values, dtype="object")
        result = casting(serie)
        assert result.dtype == "boolean"

    def test_boolean_mapping_produces_correct_values(self):
        serie = pd.Series(["oui", "non", "yes", "no"], dtype="object")
        result = casting(serie)
        assert list(result) == [True, False, True, False]

    def test_extra_non_boolean_value_prevents_boolean_cast(self):
        serie = pd.Series(["true"]*15 + ["false"]*18 + ["maybe"]*3, dtype="object")
        result = casting(serie, boolean_threshold = 0.90)
        assert result.dtype == "boolean"

    def test_boolean_detection_with_missing_values(self):
        serie = pd.Series(["oui", "non", None], dtype="object")
        result = casting(serie)
        assert result.dtype == "boolean"
        assert result.isna().sum() == 1


# ----------------------------------------------------------------------
# 4. Numeric Conversion
# ----------------------------------------------------------------------

class TestNumericConversion:

    def test_pure_integers_as_strings_become_Int64(self):
        serie = pd.Series(["1", "2", "3", "4"], name="num", dtype="object")
        result = casting(serie)
        assert result.dtype == "Int64"
        assert list(result) == [1, 2, 3, 4]

    def test_integer_like_floats_become_Int64(self):
        serie = pd.Series(["1.0", "2.0", "3.0"], name="num", dtype="object")
        result = casting(serie)
        assert result.dtype == "Int64"

    def test_decimal_values_become_float64(self):
        serie = pd.Series(["1.1", "2.2", "3.3"], name="num", dtype="object")
        result = casting(serie)
        assert result.dtype == "float64"

    def test_conversion_rate_below_threshold_is_rejected(self):
        values = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "abc"]
        serie = pd.Series(values, name="num", dtype="object")
        result = casting(serie)
        assert result.dtype not in ("Int64", "float64")

    def test_custom_lower_threshold_allows_conversion(self):
        values = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "abc"]
        serie = pd.Series(values, name="num", dtype="object")
        result = casting(serie, numeric_threshold=0.85)
        assert result.dtype in ("Int64", "float64")

    def test_conversion_rate_computed_on_non_missing_only(self):
        serie = pd.Series(["1", "2", None, "3"], name="num", dtype="object")
        result = casting(serie)
        assert result.dtype == "Int64"
        assert result.isna().sum() == 1
        assert list(result.dropna()) == [1, 2, 3]


# ----------------------------------------------------------------------
# 5. Datetime Conversion
# ----------------------------------------------------------------------

class TestDatetimeConversion:

    def test_valid_dates_are_converted(self):
        values = ["2023-01-01", "2023-02-01", "2023-03-01"]
        serie = pd.Series(values, name="date_col", dtype="object")
        result = casting(serie)
        assert pd.api.types.is_datetime64_any_dtype(result)

    def test_conversion_rate_below_threshold_is_rejected(self):
        values = ["2023-01-01", "2023-02-01", "not_a_date", "also_not"]
        serie = pd.Series(values, name="date_col", dtype="object")
        result = casting(serie)
        assert not pd.api.types.is_datetime64_any_dtype(result)

    def test_custom_lower_threshold_allows_conversion(self):
        values = ["2023-01-01", "2023-02-01", "not_a_date"]
        serie = pd.Series(values, name="date_col", dtype="object")
        result = casting(serie, datetime_threshold=0.6)
        assert pd.api.types.is_datetime64_any_dtype(result)


# ----------------------------------------------------------------------
# 6. Category vs String (cardinality)
# ----------------------------------------------------------------------

class TestCategoryVsString:

    def test_low_absolute_cardinality_becomes_category(self):
        values = ["red", "blue", "green"] * 5
        serie = pd.Series(values, name="color", dtype="object")
        result = casting(serie)
        assert isinstance(result.dtype, pd.CategoricalDtype)

    def test_low_relative_cardinality_becomes_category(self):
        
        values = ["A"] * 15 + ["B"] * 8 + ["C"] * 11
        serie = pd.Series(values, name="grp", dtype="object")
        result = casting(serie)
        assert isinstance(result.dtype, pd.CategoricalDtype)

    def test_high_cardinality_becomes_string(self):
        values = [f"user_{i}" for i in range(100)]
        serie = pd.Series(values, name="user_id", dtype="object")
        result = casting(serie)
        assert result.dtype == "string"

    def test_custom_absolute_cardinality_threshold(self):
        values = ["a", "b", "c", "d", "e"]
        s = pd.Series(values, name="col", dtype="object")
        result = casting(s, cardinality_absolute=3)
        assert result.dtype == "string"

    def test_custom_relative_cardinality_threshold(self):
        values = [f"v{i}" for i in range(20)]
        serie = pd.Series(values, name="col", dtype="object")
        result = casting(serie, cardinality_relative=1.5, cardinality_absolute=0)
        assert isinstance(result.dtype, pd.CategoricalDtype)


# ----------------------------------------------------------------------
# 7. Others types not handled
# ----------------------------------------------------------------------

class TestOtherDtypes:

    def test_complex_dtype_returned_unchanged(self):
        serie = pd.Series([1 + 2j, 3 + 4j], name="complex_col")
        result = casting(serie)
        pd.testing.assert_series_equal(result, serie)


# ----------------------------------------------------------------------
# 8. Immuability of series 'name' and original data
# ----------------------------------------------------------------------

class TestCrossCutting:

    def test_variable_name_is_preserved_after_numeric_cast(self):
        serie = pd.Series(["1", "2", "3"], name="my_var", dtype="object")
        result = casting(serie)
        assert result.name == "my_var"

    def test_variable_name_is_preserved_after_category_cast(self):
        serie = pd.Series(["a", "b", "a"], name="my_cat", dtype="object")
        result = casting(serie)
        assert result.name == "my_cat"

    def test_function_does_not_mutate_input_series(self):
        serie = pd.Series(["1", "2", "3"], name="num", dtype="object")
        serie_copy = serie.copy()
        casting(serie)
        pd.testing.assert_series_equal(serie, serie_copy)
