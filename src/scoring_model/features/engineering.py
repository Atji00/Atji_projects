import pandas as pd


class FeatureEngineer:

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data.copy()

    def transform(self) -> pd.DataFrame:

        self.data["total_contacts"] = self.data["campaign"] + self.data["previous"]

        self.data["balance_per_contact"] = (
                                                self.data["balance"]
                                                / self.data["total_contacts"].replace(0, 1)
                                            )

        return self.data



