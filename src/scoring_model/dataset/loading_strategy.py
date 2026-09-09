from pathlib import Path
from typing import Any

import pandas as pd

from scoring_model.runtime.config import ConfigLoader


class Strategy:
    """load configuration from data.yaml for loading strategy"""

    def __init__(self, config_filename: str = "data.yaml") -> None:

        self.data_config = ConfigLoader().load_yaml(config_filename)

        raw_config = self.data_config["data"]["raw"]

        self.formats_config = raw_config["formats"]
        
        self.loading_config = raw_config["loading"]

        self.max_file_size_mb = self.loading_config["max_file_size_mb"]

        self.chunksize = self.loading_config["chunksize"]

    def load(self, path: Path) -> pd.DataFrame:

        """Load a raw file according to its format and size."""

        extension = (path.suffix.lower().lstrip("."))

        if extension not in self.formats_config:
            raise ValueError(
                f"Unsupported file format: .{extension}. "
                f"Supported formats: "
                f"{list(self.formats_config)}"
            )

        file_config = self.formats_config[extension]

        read_options: dict[str, Any] = {
            "sep": file_config["separator"],
            "encoding": file_config["encoding"]
        }

        size_mb = path.stat().st_size / (1024 ** 2)

        if size_mb >= self.max_file_size_mb:
            read_options["chunksize"] = self.chunksize

        return pd.read_csv(path, **read_options)
