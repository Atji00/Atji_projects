import pandas as pd
import pytest

import scoring_model.dataset.splitting as splitting_module
from scoring_model.dataset.splitting import TrainTestSplitter

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def splitter(patch_paths):
    patch_paths(splitting_module)
    return TrainTestSplitter()


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "age": range(100),
            "job": ["admin.", "technician"] * 50,
            "y": [True] * 20 + [False] * 80,
        }
    )


# ----------------------------------------------------------------------
# 1. Split
# ----------------------------------------------------------------------

class TestSplit:

    def test_sizes_follow_test_size(self, splitter, data):
        X_train, X_test, y_train, y_test = splitter.split(data)
        assert len(X_train) == len(y_train) == 80
        assert len(X_test) == len(y_test) == 20

    def test_target_is_removed_from_features(self, splitter, data):
        X_train, X_test, _, _ = splitter.split(data)
        assert "y" not in X_train.columns
        assert "y" not in X_test.columns

    def test_split_is_stratified(self, splitter, data):
        _, _, y_train, y_test = splitter.split(data)
        assert y_train.mean() == pytest.approx(0.20)
        assert y_test.mean() == pytest.approx(0.20)

    def test_split_is_reproducible(self, splitter, data):
        X_train_1, *_ = splitter.split(data)
        X_train_2, *_ = splitter.split(data)
        pd.testing.assert_frame_equal(X_train_1, X_train_2)

    def test_no_row_is_lost_or_duplicated(self, splitter, data):
        X_train, X_test, _, _ = splitter.split(data)
        assert sorted(X_train.index.tolist() + X_test.index.tolist()) == list(range(100))

    def test_missing_target_raises(self, splitter, data):
        with pytest.raises(KeyError):
            splitter.split(data.drop(columns="y"))


# ----------------------------------------------------------------------
# 2. Saved files
# ----------------------------------------------------------------------

class TestSavedFiles:

    def test_parquet_files_are_written(self, splitter, data, tmp_project):
        splitter.split(data)
        train_dir = tmp_project.intermediate / "unprocessed_train"
        test_dir = tmp_project.intermediate / "unprocessed_test"
        assert (train_dir / "X_train_unprocessed.parquet").is_file()
        assert (train_dir / "y_train.parquet").is_file()
        assert (test_dir / "X_test_unprocessed.parquet").is_file()
        assert (test_dir / "y_test.parquet").is_file()

    def test_saved_target_keeps_target_name(self, splitter, data, tmp_project):
        _, _, _, y_test = splitter.split(data)
        saved = pd.read_parquet(tmp_project.intermediate / "unprocessed_test" / "y_test.parquet")
        assert list(saved.columns) == ["y"]
        assert saved["y"].tolist() == y_test.tolist()
