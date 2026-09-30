import pandas as pd
import pytest

from scoring_model.features.engineering import FeatureEngineer

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "balance": [1000, -500, 300],
            "campaign": [1, 2, 0],
            "previous": [3, 0, 0],
        }
    )


# ----------------------------------------------------------------------
# 1. New features
# ----------------------------------------------------------------------

class TestComputedFeatures:

    def test_new_columns_are_added(self, data):
        result = FeatureEngineer(data).compute()
        assert list(result.columns) == [
                                        "balance", "campaign", "previous",
                                        "total_contacts", "balance_per_contact",
                                        ]

    def test_total_contacts_is_campaign_plus_previous(self, data):
        result = FeatureEngineer(data).compute()
        assert list(result["total_contacts"]) == [4, 2, 0]

    def test_balance_per_contact(self, data):
        result = FeatureEngineer(data).compute()
        assert result["balance_per_contact"].iloc[0] == pytest.approx(250.0)
        assert result["balance_per_contact"].iloc[1] == pytest.approx(-250.0)

    def test_zero_contacts_divides_by_one(self, data):
        result = FeatureEngineer(data).compute()
        assert result["balance_per_contact"].iloc[2] == pytest.approx(300.0)


# ----------------------------------------------------------------------
# 2. Cross cutting
# ----------------------------------------------------------------------

class TestCrossCutting:

    def test_input_data_is_not_mutated(self, data):
        data_copy = data.copy()
        FeatureEngineer(data).compute()
        pd.testing.assert_frame_equal(data, data_copy)

    def test_index_is_preserved(self, data):
        data.index = ["a", "b", "c"]
        result = FeatureEngineer(data).compute()
        assert list(result.index) == ["a", "b", "c"]

    def test_missing_column_raises(self, data):
        with pytest.raises(KeyError):
            FeatureEngineer(data.drop(columns="previous")).compute()
