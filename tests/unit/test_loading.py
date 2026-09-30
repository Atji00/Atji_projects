import pandas as pd
import pytest

import scoring_model.dataset.loading as loading_module
from scoring_model.dataset.loading import RawDataLoader

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def loader(patch_paths):

    paths = patch_paths(loading_module)

    (paths.raw / "bank.csv").write_text("age;job\n25;admin.\n", encoding="utf-8")
    (paths.raw / "bank.txt").write_text("age\tjob\n25\tadmin.\n", encoding="utf-8")
    (paths.raw / "notes.md").write_text("not a dataset", encoding="utf-8")

    return RawDataLoader()


# ----------------------------------------------------------------------
# 1. Load a raw file
# ----------------------------------------------------------------------

class TestLoad:

    def test_existing_file_is_loaded(self, loader):
        result = loader.load("bank.csv")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 2)

    def test_missing_file_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load("missing.csv")

    def test_unsupported_format_raises(self, loader):
        with pytest.raises(ValueError):
            loader.load("notes.md")


# ----------------------------------------------------------------------
# 2. Available files
# ----------------------------------------------------------------------

class TestAvailableFiles:

    def test_lists_only_supported_formats(self, loader):
        assert loader.available_files() == ["bank.csv", "bank.txt"]

    def test_missing_raw_directory_raises(self, loader, tmp_project):
        for file in tmp_project.raw.iterdir():
            file.unlink()
        tmp_project.raw.rmdir()
        with pytest.raises(FileNotFoundError):
            loader.available_files()
