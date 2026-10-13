import base64
import io
from functools import lru_cache

import joblib
import pandas as pd
import shap
from fastapi import HTTPException
from matplotlib import pyplot as plt

from api.modules.pydantic_scheme import ClientInput, PredictionResponse
from scoring_model.eval_artefacts_func.model_interpretability import ModelInterpreter
from scoring_model.features.selected_features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from scoring_model.runtime.paths import ProjectPaths


class Artefacts(ModelInterpreter):

    def __init__(self, client: ClientInput, config_file: str = "config.yaml") -> None:

        super().__init__(config_file)

        self.paths = ProjectPaths()
        self.client = client
        
        self.model = self.paths.models_latest / "random_forest_v001.joblib"
        self.processor = self.paths.features / "processor_fitted.joblib"
        self.model_name = self.model.stem
        
        self.artefacts = self._load_artifacts()
        self.input_data = self._client_processed_data()
    
    # private method to load model and processor artifacts with caching
    @lru_cache(maxsize=1)  # noqa: B019
    def _load_artifacts(self) -> tuple:

        return joblib.load(self.model), joblib.load(self.processor)


    # Method to process client input data
    def _client_processed_data(self) -> pd.DataFrame:

        _, processor = self.artefacts

        data = self.client.model_dump(mode="json", 
                                 include=set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
                                 )

        try:

            data_processed = pd.DataFrame(
                processor.transform(pd.DataFrame([data])),
                columns=processor.get_feature_names_out()
            )

        except ValueError as error:

            raise HTTPException(status_code=500, 
                                detail=f"Model/processor mismatch: {error}"
                            )

        return data_processed

    
    # Method to plot waterfall chart for SHAP values of a client's prediction
    def plot_waterfall(self) -> str|None:

        client_name = getattr(self.client, "name", None) or "Clientxxx"

        model, _ = self.artefacts

        if self.model_name.startswith("logistic_regression"):

            explainer = shap.LinearExplainer(
                                            model,
                                            self.X_test_processed[:100]
                                            )

        else:

            explainer = shap.TreeExplainer(model)

        shap_explanation = explainer(self.input_data)[0]

        shap_explanation.feature_names = self._clean_feature_names(
                                        self.input_data.columns
                                        )

        # Les modeles a arbres renvoient une explication par classe :
        # on garde celle de la classe positive (defaut)
        if shap_explanation.values.ndim == 2:

            shap_explanation = shap_explanation[:, 1]


        probability = model.predict_proba(self.input_data)[0, 1]

        self._new_figure()

        shap.plots.waterfall(shap_explanation, max_display=10, show=False)

        fig = plt.gcf()

        try:

            fig.suptitle(f"Waterfall_Plot pour Client: {client_name.upper()} | "
                         f"Prob: {round(probability,4)}")

            fig.text(
                    0.5,
                    -0.08,
                    "SHAP values in log-odds (model output before sigmoid)",
                    ha="center",
                    fontsize=9,
                    color="grey",
                    )

            with io.BytesIO() as image_buffer:

                fig.savefig(
                                image_buffer,
                                format="png",
                                dpi=300,
                                bbox_inches="tight",
                            )

                # image in base64 for HTML injection
                image = image_buffer.getvalue()

                return base64.b64encode(image).decode("ascii")

        except (HTTPException, ValueError, KeyError):

            return None
        
        finally:

            plt.close(fig)


    # Method to run prediction for a given client input
    def run_prediction(self) -> PredictionResponse:

        model, _ = self.artefacts

        prediction = int(model.predict(self.input_data)[0])
        
        probability = float(model.predict_proba(self.input_data)[0, 1].round(4))


        return PredictionResponse(prediction=prediction, probability=probability)