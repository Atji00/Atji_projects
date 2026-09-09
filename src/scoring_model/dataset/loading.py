from pathlib import Path

import pandas as pd

from scoring_model.runtime.config import ConfigLoader
from scoring_model.runtime.paths import ProjectPaths

from .loading_strategy import Strategy


class RawDataLoader:

    """\nload raw datasets."""

    def __init__(self, config_filename: str = "data.yaml") -> None:
      
        self.config = ConfigLoader().load_yaml(config_filename)

        self.paths = ProjectPaths()

        self.raw_config = self.config["data"]["raw"]

        self.formats_config = self.raw_config["formats"]

        self.strategy = Strategy(config_filename)

    def available_files(self) -> list[str]:
     
        if not self.paths.raw_data.is_dir():

            raise FileNotFoundError(
                f"Raw data directory not found: "
                f"{self.paths.raw_data}"
            )

        allowed_formats = {format.lower().lstrip(".")
                            for format in self.formats_config
                        }

        return sorted(
                        path.name
                        for path in self.paths.raw_data.iterdir()
                        if (
                            path.is_file()
                            and path.suffix.lower().lstrip(".")
                            in allowed_formats
                        )
                    )

    def load(self, filename: str) -> pd.DataFrame:

        path = self.paths.raw_data / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Raw data file not found: {path}"
            )

        return self.strategy.load(path)