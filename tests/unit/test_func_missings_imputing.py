import numpy as np
import pandas as pd
import pytest

from scoring_model.utils.func_missings_imputing import impute_missings

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def numeric_df():
    return pd.DataFrame({
        "age": [25.0, np.nan, 35.0, 45.0, np.nan],
        "income": [1000.0, 2000.0, np.nan, 4000.0, 5000.0],
    })


@pytest.fixture
def categorical_df():
    return pd.DataFrame({
        "city": ["Paris", None, "Lyon", "Paris", "Lyon"],
        "job": ["dev", "dev", None, "sales", "sales"],
    })


@pytest.fixture
def mixed_df():
    return pd.DataFrame({
        "age": [25.0, np.nan, 35.0, 45.0, 22.0],
        "income": [1000.0, 2000.0, 3000.0, np.nan, 5000.0],
        "city": ["Paris", "Lyon", None, "Paris", "Lyon"],
    })


@pytest.fixture
def no_missing_df():
    return pd.DataFrame({
        "age": [25.0, 30.0, 35.0],
        "city": ["Paris", "Lyon", "Nice"],
    })


# ---------------------------------------------------------------------
# Parameters Validation
# ---------------------------------------------------------------------

class TestsMissingsImputing:
        
    def test_invalid_numeric_strategy_raises(self, numeric_df):
        with pytest.raises(ValueError):
            impute_missings(numeric_df, numeric_strategy="bad_strategy")


    def test_invalid_categorical_strategy_raises(self, categorical_df):
        with pytest.raises(ValueError):
            impute_missings(categorical_df, categorical_strategy="bad_strategy")


    def test_invalid_knn_neighbors_raises(self, numeric_df):
        with pytest.raises(ValueError):
            impute_missings(numeric_df, numeric_strategy="knn", knn_neighbors=0)


    def test_negative_knn_neighbors_raises(self, numeric_df):
        with pytest.raises(ValueError):
            impute_missings(numeric_df, numeric_strategy="knn", knn_neighbors=-3)


    # ---------------------------------------------------------------------
    # Numeric Imputing
    # ---------------------------------------------------------------------

    def test_numeric_median_imputation_fills_all_na(self, numeric_df):

        result = impute_missings(numeric_df, numeric_strategy="median")

        assert result[["age", "income"]].isna().sum().sum() == 0


    def test_numeric_median_imputation_correct_value(self, numeric_df):
        
        result = impute_missings(numeric_df, numeric_strategy="median")

        assert result.loc[1, "age"] == pytest.approx(35.0)


    def test_numeric_mean_imputation_correct_value(self, numeric_df):
        
        result = impute_missings(numeric_df, numeric_strategy="mean")

        assert result.loc[1, "age"] == pytest.approx(35.0)


    def test_numeric_knn_imputation_fills_all_na(self, numeric_df):

        result = impute_missings(numeric_df, numeric_strategy="knn", knn_neighbors=2)

        assert result[["age", "income"]].isna().sum().sum() == 0


    def test_no_numeric_columns_untouched(self, categorical_df):
        
        result = impute_missings(categorical_df, numeric_strategy="mean")

        assert result.select_dtypes(include="number").empty


    # ---------------------------------------------------------------------
    # Categorial Imputing
    # ---------------------------------------------------------------------

    def test_categorical_most_frequent_imputation(self, categorical_df):

        result = impute_missings(categorical_df, categorical_strategy="most_frequent")

        assert result["city"].isna().sum() == 0
   
        assert result.loc[1, "city"] in {"Paris", "Lyon"}


    def test_categorical_constant_imputation(self, categorical_df):

        result = impute_missings(categorical_df, categorical_strategy="constant")

        assert result.loc[1, "city"] == "missing"

        assert result.loc[2, "job"] == "missing"


    def test_categorical_knn_imputation_fills_all_na(self, categorical_df):

        result = impute_missings(categorical_df, categorical_strategy="knn", knn_neighbors=2)

        assert result["city"].isna().sum() == 0

        assert result["job"].isna().sum() == 0


    def test_categorical_knn_imputation_returns_known_categories(self, categorical_df):
        
        result = impute_missings(categorical_df, categorical_strategy="knn", knn_neighbors=2)

        assert result["city"].isin(["Paris", "Lyon"]).all()


    def test_no_categorical_columns_does_not_error(self, numeric_df):
        
        result = impute_missings(numeric_df, categorical_strategy="constant")

        assert result[["age", "income"]].isna().sum().sum() == 0


    # ---------------------------------------------------------------------
    # Missing_indicator
    # ---------------------------------------------------------------------

    def test_missing_indicator_creates_expected_columns(self, mixed_df):

        result = impute_missings(mixed_df, missing_indicator=True)

        assert "age_missing_indicator" in result.columns

        assert "income_missing_indicator" in result.columns

        assert "city_missing_indicator" in result.columns


    def test_missing_indicator_values_match_original_na_positions(self, mixed_df):

        original_na = mixed_df["age"].isna()

        result = impute_missings(mixed_df, missing_indicator=True)

        assert (result["age_missing_indicator"] == original_na.astype("int64")).all()


    def test_missing_indicator_not_created_for_complete_columns(self, mixed_df):
 
        df = mixed_df.copy()

        df["complete_col"] = ["a", "b", "c", "d", "e"]

        result = impute_missings(df, missing_indicator=True)

        assert "complete_col_missing_indicator" not in result.columns


    def test_missing_indicator_false_by_default_no_extra_columns(self, mixed_df):

        result = impute_missings(mixed_df)

        assert not any(col.endswith("_missing_indicator") for col in result.columns)


    # ---------------------------------------------------------------------
    # General behavior
    # ---------------------------------------------------------------------

    def test_original_dataframe_not_mutated(self, numeric_df):

        original = numeric_df.copy(deep=True)

        impute_missings(numeric_df, numeric_strategy="median")

        pd.testing.assert_frame_equal(numeric_df, original)


    def test_no_missing_values_dataframe_unchanged(self, no_missing_df):

        result = impute_missings(no_missing_df)

        pd.testing.assert_frame_equal(result, no_missing_df)


    def test_mixed_numeric_and_categorical_all_imputed(self, mixed_df):

        result = impute_missings(
            mixed_df,
            numeric_strategy="mean",
            categorical_strategy="most_frequent",
        )

        assert result.isna().sum().sum() == 0


    def test_returns_dataframe_with_same_shape_when_no_indicator(self, mixed_df):

        result = impute_missings(mixed_df, missing_indicator=False)

        assert result.shape == mixed_df.shape


    def test_returns_dataframe_type(self, numeric_df):

        result = impute_missings(numeric_df)

        assert isinstance(result, pd.DataFrame)