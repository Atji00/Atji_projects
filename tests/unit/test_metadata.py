from scoring_model.dataset.metadata import Variables_Description
from scoring_model.runtime.config import ConfigLoader

# ----------------------------------------------------------------------
# Consistency with data.yaml
# ----------------------------------------------------------------------

class TestVariablesDescription:

    def test_all_descriptions_are_non_empty_strings(self):
        assert all(isinstance(value, str) and value.strip()
                   for value in Variables_Description.values())

    def test_covers_every_validated_column_and_target(self):
        validation = ConfigLoader().load_yaml("data.yaml")["data"]["validation"]
        expected = set(validation["columns"]) | set(validation["target"])
        assert set(Variables_Description) == expected
