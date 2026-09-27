import json
import subprocess
import sys
import time

import mlflow
import pandas as pd
from mlflow import ActiveRun, MlflowClient, MlflowException

from scoring_model.runtime.paths import ProjectPaths
from scoring_model.scripts.predict import MODEL_NAME, model_trained


class Experiment:

    """\nManage MLflow tracking.\n"""

    def __init__(self) -> None:

        #-------Paths Settings----------------

        self.paths = ProjectPaths()

        self.report = (
                        self.paths.metrics / 
                        f"{MODEL_NAME}" /
                        f"classification_report_{MODEL_NAME}.json"
                      )
        
        self.input_path = self.paths.processed

        
        #-----Model Settings--------------------
        
        self.model_params = model_trained.get_params()

        self.input = pd.read_parquet(self.input_path / 
                                     "processed_test" / 
                                     "X_test_processed.parquet"
                                     )

        self.run_name = f"Run_{MODEL_NAME}"

        self.artefact_path = MODEL_NAME

        with self.report.open("r", encoding="utf-8") as file:

            report = json.load(file)

        self.metrics = {
                        "precision": report["True"]["precision"],
                        "recall": report["True"]["recall"],
                        "f1_score": report["True"]["f1-score"],
                        "accuracy": report["accuracy"]
                        }

        
        #-----Experiment Settings--------------------
        
        self.experiment_name = "Credit Scoring Modeling Experiment"

        self.tracking_uri = "http://127.0.0.1:8080"
               
        self.experiment_description = (
                                        """
                                        Predict Credit default score or probaility running 
                                        3 machine learning models:\n
                                        - Logistic Classifier
                                        - RandomForest Classifier
                                        - XgBoost Classifier  
                                        """
                                      )
        
        self.experiment_tags = {
                                "project_name": "Credit Scoring Modeling",
                                "Owner": "ATJI Cheick Abdoul Aziz",
                                "project_quarter": "Q4-2026",
                                "mlflow.note.content": self.experiment_description
                                }
        
        mlflow.set_tracking_uri(self.tracking_uri)

        self.client = MlflowClient(self.tracking_uri)
        
        self.server_process = self._server_starting()

        time.sleep(2)

        self.experiment_id = self._configuration()


    def _server_starting(self) -> subprocess.Popen | None:
        
        """
        Start the MLflow tracking server if it is not already running.
        """

        try:
            self.client.search_experiments()

            print(f"MLflow server already running at {self.tracking_uri}")

            return None

        except Exception:  # noqa: BLE001, S110

            pass

        print(f"Starting MLflow server at {self.tracking_uri}...")

        return subprocess.Popen(
                                    [
                                        sys.executable,
                                        "-m",
                                        "mlflow",
                                        "server",
                                        "--host",
                                        "127.0.0.1",
                                        "--port",
                                        "8080",
                                    ],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )

    # Experiment configuration
    def _configuration(self) -> str:

        """
        Return Experiment unique ID
        """

        try:

            experiment = self.client.get_experiment_by_name(self.experiment_name)

            if experiment is None:

                experiment_id = self.client.create_experiment(
                                                                name=self.experiment_name,
                                                                tags=self.experiment_tags
                                                                )

            else:

                experiment_id = experiment.experiment_id

        except MlflowException as e:

            raise RuntimeError(
                                f"Unable to configure MLflow experiment "
                                f"'{self.experiment_name}': {e}"
                                ) from e

        mlflow.set_experiment(experiment_id = experiment_id)

        return experiment_id


    # Start Run
    def start_run(self) -> ActiveRun:
                
        with mlflow.start_run(run_name=self.run_name, experiment_id=self.experiment_id) as run:

            mlflow.log_params(self.model_params)

            mlflow.log_metrics(self.metrics)

            mlflow.sklearn.log_model(
                                    sk_model=model_trained, 
                                    input_example=self.input, 
                                    artifact_path=self.artefact_path
                                    )

        print(f"MLflow run completed: {run.info.run_name}")
        
        return run
