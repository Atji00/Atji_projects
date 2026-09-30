import pandas as pd
import pytest

from scoring_model.dataset.loading_strategy import Strategy

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def strategy(patch_paths):
    patch_paths()
    return Strategy()


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("age;job\n25;admin.\n40;technician\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------

class TestConfiguration:

    def test_configuration_is_read_from_data_yaml(self, strategy):
        assert set(strategy.formats_config) == {"csv", "txt"}
        assert strategy.max_file_size_mb == 500
        assert strategy.chunksize == 100000


# ----------------------------------------------------------------------
# 2. Loading by format
# ----------------------------------------------------------------------

class TestLoadByFormat:

    def test_csv_is_read_with_semicolon_separator(self, strategy, csv_file):
        result = strategy.load(csv_file)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["age", "job"]
        assert list(result["age"]) == [25, 40]

    def test_txt_is_read_with_tab_separator(self, strategy, tmp_path):
        path = tmp_path / "sample.txt"
        path.write_text("age\tjob\n25\tadmin.\n", encoding="utf-8")
        result = strategy.load(path)
        assert list(result.columns) == ["age", "job"]

    def test_extension_is_case_insensitive(self, strategy, tmp_path):
        path = tmp_path / "SAMPLE.CSV"
        path.write_text("age;job\n25;admin.\n", encoding="utf-8")
        assert strategy.load(path).shape == (1, 2)

    @pytest.mark.parametrize("filename", ["sample.xlsx", "sample.json", "sample"])
    def test_unsupported_format_raises(self, strategy, tmp_path, filename):
        path = tmp_path / filename
        path.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            strategy.load(path)


# ----------------------------------------------------------------------
# 3. Large files are read by chunks
# ----------------------------------------------------------------------

class TestChunks:

    def test_small_file_is_read_in_one_block(self, strategy, csv_file):
        assert isinstance(strategy.load(csv_file), pd.DataFrame)

    def test_file_above_size_limit_is_read_by_chunks(self, strategy, csv_file):
        strategy.max_file_size_mb = 0
        strategy.chunksize = 1
        with strategy.load(csv_file) as reader:
            chunks = list(reader)
        assert len(chunks) == 2
        assert all(len(chunk) == 1 for chunk in chunks)
