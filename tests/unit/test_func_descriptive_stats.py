import pandas as pd
import pytest

from scoring_model.utils.func_descriptive_stats import (
                                                            category_distribution,
                                                            check_variables,
                                                            crossed_statistics,
                                                        )

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df():
    """Create a test DataFrame containing several pandas dtypes."""
    return pd.DataFrame(
        {
            "cat_col": pd.Categorical(["a", "b", "a", "c", "b", "a"]),
            "bool_col": [True, False, True, False, True, False],
            "num_col": [1, 2, 3, 4, 5, 6],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            "str_col": ["x", "y", "z", "w", "v", "u"],
        }
    )


# ---------------------------------------------------------------------------
# check_variables
# ---------------------------------------------------------------------------

class TestCheckVariables:

    def test_valid_categorical_and_numeric(self, df):
        """Check that valid categorical and numeric variables are retained."""
        catlist, statvars = check_variables(
            df,
            ["cat_col", "bool_col"],
            ["num_col", "float_col"],
        )

        assert catlist == ["cat_col", "bool_col"]
        assert statvars == ["num_col", "float_col"]

    def test_missing_categorical_variable_is_removed(self, df):
        """Check that a missing categorical variable is removed with a warning."""
        with pytest.warns(UserWarning, match="not in data columns"):
            catlist, statvars = check_variables(
                df,
                ["missing_col"],
                ["num_col"],
            )

        assert catlist == []
        assert statvars == ["num_col"]

    def test_missing_numeric_variable_is_removed(self, df):
        """Check that a missing numeric variable is removed with a warning."""
        with pytest.warns(UserWarning, match="not in data columns"):
            catlist, statvars = check_variables(
                df,
                ["cat_col"],
                ["missing_col"],
            )

        assert catlist == ["cat_col"]
        assert statvars == []

    def test_non_categorical_variable_is_removed(self, df):
        """Check that a non-categorical variable is rejected."""
        with pytest.warns(UserWarning, match="not in"):
            catlist, statvars = check_variables(
                df,
                ["str_col"],
                [],
            )

        assert catlist == []
        assert statvars == []

    def test_non_numeric_variable_is_removed(self, df):
        """Check that a non-numeric variable is rejected."""
        with pytest.warns(UserWarning, match="not in"):
            catlist, statvars = check_variables(
                df,
                [],
                ["str_col"],
            )

        assert statvars == []

    def test_boolean_variable_is_not_numeric(self, df):
        """Check that boolean variables are excluded from numeric variables."""
        with pytest.warns(UserWarning, match="not in"):
            catlist, statvars = check_variables(
                df,
                [],
                ["bool_col"],
            )

        assert catlist == []
        assert statvars == []

    def test_boolean_variable_is_categorical(self, df):
        """Check that boolean variables are accepted as categorical variables."""
        catlist, statvars = check_variables(
            df,
            ["bool_col"],
            [],
        )

        assert catlist == ["bool_col"]
        assert statvars == []

    def test_mixed_valid_and_invalid_variables(self, df):
        """Check that valid variables are retained and invalid ones removed."""
        with pytest.warns(UserWarning):
            catlist, statvars = check_variables(
                df,
                ["cat_col", "missing_col"],
                ["num_col", "str_col"],
            )

        assert catlist == ["cat_col"]
        assert statvars == ["num_col"]

    def test_empty_input_lists(self, df):
        """Check that empty input lists return empty lists."""
        catlist, statvars = check_variables(
            df,
            [],
            [],
        )

        assert catlist == []
        assert statvars == []

    def test_float_variable_is_numeric(self, df):
        """Check that floating-point variables are accepted as numeric."""
        _, statvars = check_variables(
            df,
            [],
            ["float_col"],
        )

        assert statvars == ["float_col"]


# ---------------------------------------------------------------------------
# category_distribution
# ---------------------------------------------------------------------------

class TestCategoryDistribution:

    def test_empty_list_raises_value_error(self, df):
        """Check that an empty variable list raises ValueError."""
        with pytest.raises(
            ValueError,
            match="list of variables empty",
        ):
            category_distribution(df, [])

    def test_valid_single_variable(self, df):
        """Check that a valid categorical variable produces a summary table."""
        result = category_distribution(
            df,
            ["cat_col"],
        )

        assert list(result.columns) == [
            "Variable",
            "Size of observations",
            "Percentage (%)",
        ]

        assert result.index[-1] == "Total"
        assert (result["Variable"] == "cat_col").all()

        assert (
            result.loc["Total", "Size of observations"]
            == len(df)
        )

        assert result.loc[
            "Total",
            "Percentage (%)",
        ] == pytest.approx(100.0)

    def test_counts_are_correct(self, df):
        """Check that category frequencies are correctly calculated."""
        result = category_distribution(
            df,
            ["cat_col"],
        )

        assert result.loc[
            "a",
            "Size of observations",
        ] == 3

        assert result.loc[
            "b",
            "Size of observations",
        ] == 2

        assert result.loc[
            "c",
            "Size of observations",
        ] == 1

    def test_percentages_are_correct(self, df):
        """Check that category percentages are correctly calculated."""
        result = category_distribution(
            df,
            ["cat_col"],
        )

        assert result.loc[
            "a",
            "Percentage (%)",
        ] == pytest.approx(50.0)

        assert result.loc[
            "b",
            "Percentage (%)",
        ] == pytest.approx(33.33)

        assert result.loc[
            "c",
            "Percentage (%)",
        ] == pytest.approx(16.67)

    def test_missing_column_is_caught(self, df, capsys):
        """Check that a missing variable is reported and handled."""
        with pytest.raises(ValueError):
            category_distribution(
                df,
                ["missing_col"],
            )

        captured = capsys.readouterr()

        assert "not found in data" in captured.out

    def test_non_categorical_column_is_caught(self, df, capsys):
        """Check that a non-categorical variable is rejected."""
        with pytest.raises(ValueError):
            category_distribution(
                df,
                ["num_col"],
            )

        captured = capsys.readouterr()

        assert "is not a <dtype.category or dtype.bool>" in captured.out

    def test_mixed_valid_and_invalid_variables(self, df):
        """Check that valid variables are included while invalid ones are skipped."""
        result = category_distribution(
            df,
            ["cat_col", "num_col"],
        )

        assert set(
            result["Variable"].unique()
        ) == {"cat_col"}

    def test_multiple_categorical_variables(self, df):
        """Check that several categorical variables are processed."""
        result = category_distribution(
            df,
            ["cat_col", "bool_col"],
        )

        assert set(
            result["Variable"].unique()
        ) == {
            "cat_col",
            "bool_col",
        }

    def test_total_percentage_is_100(self, df):
        """Check that the total percentage is equal to 100 percent."""
        result = category_distribution(
            df,
            ["cat_col"],
        )

        assert result.loc[
            "Total",
            "Percentage (%)",
        ] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# crossed_statistics
# ---------------------------------------------------------------------------

class TestCrossedStatistics:

    EXPECTED_COLUMNS = [
        "min",
        "max",
        "mean",
        "st_deviation",
        "quartile1",
        "median",
        "quartile3",
        "category_size",
    ]

    def test_valid_single_cat_single_statvar(self, df):
        """Check the output structure for one categorical and one numeric variable."""
        result = crossed_statistics(
            df,
            ["cat_col"],
            ["num_col"],
        )

        assert isinstance(result, dict)
        assert list(result.keys()) == ["num_col"]

        table = result["num_col"]

        assert table.index.names == [
            "Categories",
            "modalities",
        ]

        assert list(table.columns) == self.EXPECTED_COLUMNS

        assert ("cat_col", "a") in table.index

    def test_statistics_values_are_correct(self, df):
        """Check that descriptive statistics are correctly calculated."""
        result = crossed_statistics(
            df,
            ["cat_col"],
            ["num_col"],
        )

        table = result["num_col"]

        row = table.loc[
            ("cat_col", "a")
        ]

        assert row["min"] == 1
        assert row["max"] == 6
        assert row["mean"] == pytest.approx(10 / 3)
        assert row["category_size"] == 3

        assert row["median"] == 3

        assert row["quartile1"] == pytest.approx(2)
        assert row["quartile3"] == pytest.approx(4.5)

    def test_multiple_categorical_variables(self, df):
        """Check that statistics are calculated for multiple categorical variables."""
        result = crossed_statistics(
            df,
            ["cat_col", "bool_col"],
            ["num_col"],
        )

        table = result["num_col"]

        categories_present = set(
            table.index.get_level_values("Categories")
        )

        assert categories_present == {
            "cat_col",
            "bool_col",
        }

    def test_multiple_numeric_variables(self, df):
        """Check that statistics are returned for each numeric variable."""
        result = crossed_statistics(
            df,
            ["cat_col"],
            ["num_col", "float_col"],
        )

        assert set(result.keys()) == {
            "num_col",
            "float_col",
        }

        assert list(result["num_col"].columns) == self.EXPECTED_COLUMNS
        assert list(result["float_col"].columns) == self.EXPECTED_COLUMNS

    def test_empty_categorical_list_raises_value_error(self, df):
        """Check that an empty categorical list raises ValueError."""
        with pytest.raises(
            ValueError,
            match="One or both lists empty",
        ):
            crossed_statistics(
                df,
                [],
                ["num_col"],
            )

    def test_empty_numeric_list_raises_value_error(self, df):
        """Check that an empty numeric list raises ValueError."""
        with pytest.raises(
            ValueError,
            match="One or both lists empty",
        ):
            crossed_statistics(
                df,
                ["cat_col"],
                [],
            )

    def test_missing_categorical_variable_raises_value_error(self, df):
        """Check that an invalid categorical variable is filtered out."""
        with pytest.warns(
            UserWarning,
            match="not in data columns",
        ):
            with pytest.raises(
                ValueError,
                match="One or both lists empty",
            ):
                crossed_statistics(
                    df,
                    ["missing_col"],
                    ["num_col"],
                )

    def test_missing_numeric_variable_raises_value_error(self, df):
        """Check that an invalid numeric variable is filtered out."""
        with pytest.warns(
            UserWarning,
            match="not in data columns",
        ):
            with pytest.raises(
                ValueError,
                match="One or both lists empty",
            ):
                crossed_statistics(
                    df,
                    ["cat_col"],
                    ["missing_col"],
                )

    def test_invalid_numeric_variable_is_filtered(self, df):
        """Check that invalid numeric variables are removed before computation."""
        with pytest.warns(
            UserWarning,
            match="not in",
        ):
            result = crossed_statistics(
                df,
                ["cat_col"],
                ["bool_col", "num_col"],
            )

        assert list(result.keys()) == ["num_col"]

    def test_invalid_categorical_variable_is_filtered(self, df):
        """Check that invalid categorical variables are removed before computation."""
        with pytest.warns(
            UserWarning,
            match="not in",
        ):
            result = crossed_statistics(
                df,
                ["str_col", "cat_col"],
                ["num_col"],
            )

        table = result["num_col"]

        categories_present = set(
            table.index.get_level_values("Categories")
        )

        assert categories_present == {"cat_col"}

    def test_category_size_is_correct(self, df):
        """Check that the number of observations per category is correct."""
        result = crossed_statistics(
            df,
            ["cat_col"],
            ["num_col"],
        )

        table = result["num_col"]

        assert table.loc[
            ("cat_col", "a"),
            "category_size",
        ] == 3

        assert table.loc[
            ("cat_col", "b"),
            "category_size",
        ] == 2

        assert table.loc[
            ("cat_col", "c"),
            "category_size",
        ] == 1

    def test_standard_deviation_is_correct(self, df):
        """Check that the standard deviation is correctly calculated."""
        result = crossed_statistics(
            df,
            ["cat_col"],
            ["num_col"],
        )

        table = result["num_col"]

        # Values for category 'a' are [1, 3, 6].
        expected_std = pd.Series(
            [1, 3, 6]
        ).std()

        assert table.loc[
            ("cat_col", "a"),
            "st_deviation",
        ] == pytest.approx(expected_std)

    def test_boolean_categorical_variable(self, df):
        """Check that boolean variables can be used as categorical variables."""
        result = crossed_statistics(
            df,
            ["bool_col"],
            ["num_col"],
        )

        table = result["num_col"]

        categories_present = set(
            table.index.get_level_values("Categories")
        )

        assert categories_present == {"bool_col"}

        modalities_present = set(
            table.index.get_level_values("modalities")
        )

        assert modalities_present == {
            True,
            False,
        }