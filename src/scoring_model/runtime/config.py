from typing import Any

import yaml

from .paths import ProjectPaths


class ConfigLoader:
    """Load project configuration from YAML files."""

    def __init__(self) -> None:
        self.paths = ProjectPaths()
        self.config_dir = self.paths.configs

    def discover(self) -> list[str]: 
        """Return the YAML configuration existing files.""" 

        return sorted( 
            path.name for path in self.config_dir.glob("*.yaml") if path.is_file() 
            )

    def load_yaml(self, filename: str) -> dict[str, Any]:
        """Load a single YAML configuration file."""

        existing_files = self.discover()

        if not filename in existing_files:
            raise FileNotFoundError(
                f"Configuration file not found: {filename} in "
                f"Existing files: {existing_files}"
            )

        path = self.config_dir / filename
        
        with path.open(mode = "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if config is None:
            return {}

        if not isinstance(config, dict):
            raise TypeError(
                f"Configuration file must contain a dictionary: {path}"
            )

        return config

    def load(self) -> dict[str, Any]:
        """Load all YAML configuration files."""

        config: dict[str, Any] = {}

        for filename in self.discover():
            file_config = self.load_yaml(filename)
            config.update(file_config)

        return config
