import shutil

import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import scoring_model.runtime.config as config_module
from scoring_model.runtime.paths import ProjectPaths

# Non interactive backend: figures are only saved, never displayed
matplotlib.use("Agg")



@pytest.fixture
def test_data():
    return pd.DataFrame(
        {
            "age": [25, None, 35, 27,19, None],
            "job": ["admin", "technician", None, "dentist","professor", "banker"]
        }
    )


@pytest.fixture
def test_metadata_1():
    return {
            "age": "Age of the client",
            "job": "Category of jobs",
            }


@pytest.fixture
def test_metadata_2():
    return {
            "age": "Age of the client",
            "status": "Jobs status: unemployed, employed, sick, etc...",
            "salary": "Salary before "
            }


# ----------------------------------------------------------------------
# Temporary project (isolated from the real data / models / reports)
# ----------------------------------------------------------------------

REAL_PATHS = ProjectPaths()


@pytest.fixture
def tmp_project(tmp_path):

    """ProjectPaths pointing to a temporary copy of the project (configs copied)."""

    shutil.copytree(REAL_PATHS.configs, tmp_path / "configs")

    paths = ProjectPaths(tmp_path)

    for directory in (paths.raw, paths.features, paths.models_registry):
        directory.mkdir(parents=True, exist_ok=True)

    return paths


@pytest.fixture
def patch_paths(monkeypatch, tmp_project):

    """Redirect ProjectPaths() of the given modules (and ConfigLoader) to tmp_project."""

    def _patch(*modules):

        for module in (config_module, *modules):
            monkeypatch.setattr(module, "ProjectPaths", lambda *args, **kwargs: tmp_project)

        return tmp_project

    return _patch


# ----------------------------------------------------------------------
# Small processed dataset and fitted model for evaluation artefacts
# ----------------------------------------------------------------------

@pytest.fixture
def processed_data():

    rng = np.random.default_rng(42)

    n_rows = 120

    X = pd.DataFrame(
        {
            "numerical__balance": rng.normal(size=n_rows),
            "numerical__campaign": rng.normal(size=n_rows),
            "categorical__housing_1.0": rng.integers(0, 2, size=n_rows).astype(float),
        }
    )

    y = pd.Series([i % 3 == 0 for i in range(n_rows)], name="y")

    return X, y


@pytest.fixture
def fitted_model(processed_data):

    X, y = processed_data

    return LogisticRegression().fit(X, y)


@pytest.fixture
def eval_files(tmp_project, processed_data, fitted_model):

    """Write train/test parquet files expected by the evaluation classes."""

    X, y = processed_data

    processed_test = tmp_project.processed / "processed_test"
    processed_train = tmp_project.processed / "processed_train"
    unprocessed_test = tmp_project.intermediate / "unprocessed_test"
    unprocessed_train = tmp_project.intermediate / "unprocessed_train"

    for directory in (processed_test, processed_train, unprocessed_test, unprocessed_train):
        directory.mkdir(parents=True, exist_ok=True)

    y_pred = pd.Series(fitted_model.predict(X), name="y_pred")
    y_prob = pd.Series(fitted_model.predict_proba(X)[:, 1], name="y_prob")

    X.to_parquet(processed_test / "X_test_processed.parquet", index=False)
    X.to_parquet(processed_train / "X_train_processed.parquet", index=False)

    y.to_frame().to_parquet(unprocessed_test / "y_test.parquet", index=False)
    y.to_frame().to_parquet(unprocessed_train / "y_train.parquet", index=False)

    y_pred.to_frame().to_parquet(processed_test / "y_predicted.parquet", index=False)
    y_prob.to_frame().to_parquet(processed_test / "y_probability.parquet", index=False)

    return {"y_test": y, "y_pred": y_pred, "y_prob": y_prob, "X": X}


@pytest.fixture
def report_dict(eval_files):

    return classification_report(eval_files["y_test"], eval_files["y_pred"], output_dict=True)
