import shutil

import joblib
import pytest
import yaml

import scoring_model.dataset.loading as loading_module
import scoring_model.dataset.splitting as splitting_module
import scoring_model.eval_artefacts_func.classification_report as classification_report_module
import scoring_model.eval_artefacts_func.confusion_matrix as confusion_matrix_module
import scoring_model.eval_artefacts_func.curve_precision_recall as precision_recall_module
import scoring_model.eval_artefacts_func.eval_report as eval_report_module
import scoring_model.eval_artefacts_func.model_interpretability as interpretability_module
import scoring_model.eval_artefacts_func.roc_curve as roc_curve_module
import scoring_model.eval_artefacts_func.train_validation as train_validation_module
import scoring_model.features.preprocessing as preprocessing_module
import scoring_model.models.models_loader as models_loader_module
import scoring_model.runtime.config as config_module
import scoring_model.scripts.predict as predict_module
from scoring_model.runtime.paths import ProjectPaths
from scoring_model.scripts.eval import eval_orchestration
from scoring_model.scripts.predict import predict
from scoring_model.scripts.train import train_orchestration

REAL_PATHS = ProjectPaths()

RAW_FILENAME = "bank.csv"

MODEL = "logistic_regression"
VERSION = "v001"
MODEL_NAME = f"{MODEL}_{VERSION}"

# Modules instantiating ProjectPaths() at runtime
PATHS_MODULES = (
                    config_module,
                    loading_module,
                    splitting_module,
                    preprocessing_module,
                    models_loader_module,
                    classification_report_module,
                    confusion_matrix_module,
                    precision_recall_module,
                    roc_curve_module,
                    train_validation_module,
                    interpretability_module,
                    eval_report_module,
                )

# Modules importing MODEL_NAME / model_trained from scripts.predict
MODEL_MODULES = (
                    predict_module,
                    classification_report_module,
                    confusion_matrix_module,
                    precision_recall_module,
                    roc_curve_module,
                    train_validation_module,
                    interpretability_module,
                    eval_report_module,
                )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def build_project(root, raw_filename: str = RAW_FILENAME) -> ProjectPaths:

    """Temporary project: real configs + real raw dataset, everything else empty."""

    shutil.copytree(REAL_PATHS.configs, root / "configs")

    paths = ProjectPaths(root)

    for directory in (paths.raw, paths.features, paths.models_registry):
        directory.mkdir(parents=True, exist_ok=True)

    if raw_filename is not None:
        shutil.copy(REAL_PATHS.raw / raw_filename, paths.raw / raw_filename)

    # Smaller ensembles to keep the integration tests fast
    models_config_path = paths.configs / "models.yaml"
    models_config = yaml.safe_load(models_config_path.read_text(encoding="utf-8"))
    models_config["random_forest"]["parameters"]["n_estimators"] = 20
    models_config["xgboost"]["parameters"]["n_estimators"] = 20
    models_config_path.write_text(yaml.safe_dump(models_config), encoding="utf-8")

    return paths


def redirect_project(monkeypatch, paths: ProjectPaths) -> None:

    """Redirect every pipeline module to the temporary project."""

    for module in PATHS_MODULES:
        monkeypatch.setattr(module, "ProjectPaths", lambda *args, **kwargs: paths)

    # predict.py builds its ProjectPaths at import time
    monkeypatch.setattr(predict_module, "paths", paths)


def use_model(monkeypatch, paths: ProjectPaths, model_name: str = MODEL_NAME):

    """Plug the model trained in the temporary registry into predict / eval modules."""

    model = joblib.load(paths.models_registry / f"{model_name}.joblib")

    for module in MODEL_MODULES:

        monkeypatch.setattr(module, "MODEL_NAME", model_name)

        if hasattr(module, "model_trained"):
            monkeypatch.setattr(module, "model_trained", model)

    return model


# ----------------------------------------------------------------------
# Pipelines executed once per test module (shared temporary project)
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def module_monkeypatch():
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield monkeypatch


@pytest.fixture(scope="module")
def project(tmp_path_factory, module_monkeypatch):

    paths = build_project(tmp_path_factory.mktemp("project"))

    redirect_project(module_monkeypatch, paths)

    return paths


@pytest.fixture(scope="module")
def trained_project(project, module_monkeypatch):

    train_orchestration(MODEL, VERSION)

    use_model(module_monkeypatch, project)

    return project


@pytest.fixture(scope="module")
def predicted_project(trained_project):

    predict()

    return trained_project


@pytest.fixture(scope="module")
def evaluated_project(predicted_project):

    eval_orchestration()

    return predicted_project


# ----------------------------------------------------------------------
# Isolated project for a single test (failure scenarios)
# ----------------------------------------------------------------------

@pytest.fixture
def empty_project(tmp_path, monkeypatch):

    """Temporary project without raw data."""

    paths = build_project(tmp_path, raw_filename=None)

    redirect_project(monkeypatch, paths)

    return paths
