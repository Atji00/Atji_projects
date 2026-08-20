import pandas as pd
import pytest


@pytest.fixture
def test_data():
    return pd.DataFrame(
        {
            "age": [25, None, 35, 27,19, None],
            "job": ["admin", "technician", None, "dentist","professor", "banker"]
        }
    )


@pytest.fixture
def test_metadata_1():
    return {
            "age": "Age of the client",
            "job": "Category of jobs",
            }


@pytest.fixture
def test_metadata_2():
    return {
            "age": "Age of the client",
            "status": "Jobs status: unemployed, employed, sick, etc...",
            "salary": "Salary before "
            }
