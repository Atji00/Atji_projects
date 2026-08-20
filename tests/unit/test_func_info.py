import pandas as pd
import pytest

from scoring_model.utils.func_info import information


# Test function for return (Pd.DataFrame)
def test_func_return(test_data, test_metadata_1):
 
    result = information(test_data, test_metadata_1)

    assert isinstance(result, pd.DataFrame)

    assert list(result["Variables"]) == ["age", "job"]

    assert list(result["Unique_values"]) == [4, 5]

    assert list(result["NA_counts"]) == [2, 1]

    assert list(result["NA_percent%"]) == pytest.approx([33.33, 16.67])

    assert list(result["Description_des_variables"]) == [
                                                            "Age of the client",
                                                            "Category of jobs"
                                                        ]

# Test function for raise ValueError
def test_func_raises_error(test_data, test_metadata_2):

    with pytest.raises(ValueError):
        information(test_data, test_metadata_2)
