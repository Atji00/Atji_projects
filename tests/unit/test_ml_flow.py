import json
import socket
from unittest.mock import MagicMock

import pytest
from mlflow import MlflowException

import scoring_model.eval_artefacts_func.ml_flow as ml_flow_module
from scoring_model.eval_artefacts_func.ml_flow import Experiment

# Real method, kept before the fixtures replace it by a stub
SERVER_IS_REACHABLE = Experiment._server_is_reachable

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def mlflow_mock(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(ml_flow_module, "mlflow", mock)
    return mock


@pytest.fixture
def client_mock(monkeypatch):
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="123")
    monkeypatch.setattr(ml_flow_module, "MlflowClient", MagicMock(return_value=client))
    return client


@pytest.fixture
def setup(monkeypatch, patch_paths, eval_files, report_dict, fitted_model, mlflow_mock, client_mock):

    paths = patch_paths(ml_flow_module)
    monkeypatch.setattr(ml_flow_module, "MODEL_NAME", "test_model")
    monkeypatch.setattr(ml_flow_module, "model_trained", fitted_model)
    monkeypatch.setattr(Experiment, "_server_is_reachable", lambda self, timeout=1.0: True)

    metrics_dir = paths.metrics / "test_model"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "classification_report_test_model.json").write_text(
                                                                        json.dumps(report_dict),
                                                                        encoding="utf-8"
                                                                        )


@pytest.fixture
def experiment(setup):
    return Experiment()


# ----------------------------------------------------------------------
# 1. Initialisation
# ----------------------------------------------------------------------

class TestInitialisation:

    def test_metrics_are_read_from_classification_report(self, experiment, report_dict):
        assert experiment.metrics == {
                                        "precision": report_dict["True"]["precision"],
                                        "recall": report_dict["True"]["recall"],
                                        "f1_score": report_dict["True"]["f1-score"],
                                        "accuracy": report_dict["accuracy"],
                                    }

    def test_model_parameters_are_read(self, experiment, fitted_model):
        assert experiment.model_params == fitted_model.get_params()

    def test_run_name_uses_model_name(self, experiment):
        assert experiment.run_name == "Run_test_model"

    def test_tracking_uri_is_set(self, experiment, mlflow_mock):
        mlflow_mock.set_tracking_uri.assert_called_once_with("http://127.0.0.1:8080")

    def test_unreachable_server_raises(self, setup, monkeypatch):
        monkeypatch.setattr(Experiment, "_server_is_reachable", lambda self, timeout=1.0: False)
        with pytest.raises(RuntimeError):
            Experiment()

    def test_missing_classification_report_raises(self, setup, tmp_project):
        (tmp_project.metrics / "test_model" / "classification_report_test_model.json").unlink()
        with pytest.raises(FileNotFoundError):
            Experiment()


# ----------------------------------------------------------------------
# 2. Server reachability
# ----------------------------------------------------------------------

class TestServerIsReachable:

    def test_listening_port_is_reachable(self, experiment):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            experiment.host, experiment.port = server.getsockname()
            assert SERVER_IS_REACHABLE(experiment) is True

    def test_closed_port_is_not_reachable(self, experiment):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            experiment.host, experiment.port = server.getsockname()
        assert SERVER_IS_REACHABLE(experiment, timeout=0.2) is False


# ----------------------------------------------------------------------
# 3. Experiment configuration
# ----------------------------------------------------------------------

class TestConfiguration:

    def test_existing_experiment_id_is_returned(self, experiment, client_mock):
        assert experiment.experiment_id == "123"
        client_mock.create_experiment.assert_not_called()

    def test_missing_experiment_is_created(self, setup, client_mock):
        client_mock.get_experiment_by_name.return_value = None
        client_mock.create_experiment.return_value = "456"
        experiment = Experiment()
        assert experiment.experiment_id == "456"
        assert client_mock.create_experiment.call_args.kwargs["name"] == experiment.experiment_name

    def test_experiment_is_activated(self, experiment, mlflow_mock):
        mlflow_mock.set_experiment.assert_called_once_with(experiment_id="123")

    def test_mlflow_error_raises_runtime_error(self, setup, client_mock):
        client_mock.get_experiment_by_name.side_effect = MlflowException("boom")
        with pytest.raises(RuntimeError):
            Experiment()


# ----------------------------------------------------------------------
# 4. Run
# ----------------------------------------------------------------------

class TestStartRun:

    def test_run_is_started_in_experiment(self, experiment, mlflow_mock):
        experiment.start_run()
        mlflow_mock.start_run.assert_called_once_with(run_name="Run_test_model", experiment_id="123")

    def test_params_and_metrics_are_logged(self, experiment, mlflow_mock):
        experiment.start_run()
        mlflow_mock.log_params.assert_called_once_with(experiment.model_params)
        mlflow_mock.log_metrics.assert_called_once_with(experiment.metrics)

    def test_model_is_logged(self, experiment, mlflow_mock, fitted_model):
        experiment.start_run()
        kwargs = mlflow_mock.sklearn.log_model.call_args.kwargs
        assert kwargs["sk_model"] is fitted_model
        assert kwargs["artifact_path"] == "test_model"

    def test_returns_active_run(self, experiment, mlflow_mock):
        run = experiment.start_run()
        assert run is mlflow_mock.start_run.return_value.__enter__.return_value
