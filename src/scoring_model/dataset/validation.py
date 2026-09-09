import pandas as pd
import pandera.pandas as pa

from scoring_model.runtime.config import ConfigLoader


class DataValidator:

    """\nValidate raw data using Pandera and data.yaml parameters.\n"""

    def __init__(self, config_filename: str = "data.yaml") -> None:

        self.config = ConfigLoader().load_yaml(config_filename)

        self.validation_config = self.config["data"]["validation"]

        self.schema = self._build_schema() 

    # <_build_schema> is a private method
    def _build_schema(self) -> pa.DataFrameSchema:

        # Initializing an empty schema
        schema: dict[str, pa.Column] = {}

        for section, content in self.validation_config.items():

            checks = []

            if section == "target":

                target_name, target_params = next(iter(content.items()))

                checks.append(pa.Check.isin(target_params["allowed"]))

                schema[target_name] = pa.Column(
                                                dtype=target_params["type"],
                                                nullable=target_params["nullable"],
                                                checks=checks
                                                )

            elif section == "columns":

                for col_name, col_params in content.items():

                    checks = []

                    if "min" in col_params:
                        checks.append(pa.Check.ge(col_params["min"]))

                    if "max" in col_params:
                        checks.append(pa.Check.le(col_params["max"]))

                    if "allowed" in col_params:
                        checks.append(pa.Check.isin(col_params["allowed"]))

                    schema[col_name] = pa.Column(
                                                dtype=col_params["type"],
                                                nullable=col_params["nullable"],
                                                checks=checks,
                                                )

        

        return pa.DataFrameSchema(columns=schema, strict=True, coerce=False)

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        
        return self.schema.validate(data)