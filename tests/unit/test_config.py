import pytest

import scoring_model.runtime.config as config_module
from scoring_model.runtime.config import ConfigLoader
from scoring_model.runtime.paths import ProjectPaths

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def config_dir(tmp_path, monkeypatch):

    paths = ProjectPaths(tmp_path)
    paths.configs.mkdir()

    monkeypatch.setattr(config_module, "ProjectPaths", lambda *args, **kwargs: paths)

    return paths.configs


@pytest.fixture
def loader(config_dir):

    (config_dir / "a.yaml").write_text("rd_seed: 42\nname: first\n", encoding="utf-8")
    (config_dir / "b.yaml").write_text("name: second\nextra: true\n", encoding="utf-8")

    return ConfigLoader()


# ----------------------------------------------------------------------
# 1. Discover
# ----------------------------------------------------------------------

class TestDiscover:

    def test_returns_sorted_yaml_files(self, loader):
        assert loader.discover() == ["a.yaml", "b.yaml"]

    def test_ignores_non_yaml_files_and_directories(self, loader, config_dir):
        (config_dir / "notes.txt").write_text("x", encoding="utf-8")
        (config_dir / "old.yml").write_text("x: 1", encoding="utf-8")
        (config_dir / "folder.yaml").mkdir()
        assert loader.discover() == ["a.yaml", "b.yaml"]

    def test_empty_config_directory(self, config_dir):
        assert ConfigLoader().discover() == []


# ----------------------------------------------------------------------
# 2. Load a single YAML file
# ----------------------------------------------------------------------

class TestLoadYaml:

    def test_returns_dictionary(self, loader):
        assert loader.load_yaml("a.yaml") == {"rd_seed": 42, "name": "first"}

    def test_missing_file_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_yaml("missing.yaml")

    def test_empty_file_returns_empty_dict(self, loader, config_dir):
        (config_dir / "empty.yaml").write_text("", encoding="utf-8")
        assert loader.load_yaml("empty.yaml") == {}

    def test_non_dictionary_content_raises(self, loader, config_dir):
        (config_dir / "list.yaml").write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(TypeError):
            loader.load_yaml("list.yaml")


# ----------------------------------------------------------------------
# 3. Load all YAML files
# ----------------------------------------------------------------------

class TestLoadAll:

    def test_merges_all_files(self, loader):
        config = loader.load()
        assert config["rd_seed"] == 42
        assert config["extra"] is True

    def test_last_file_in_alphabetical_order_wins(self, loader):
        assert loader.load()["name"] == "second"

    def test_real_project_configuration_is_loadable(self):
        config = ConfigLoader().load()
        assert config["rd_seed"] == 42
        assert "data" in config
        assert "training" in config
