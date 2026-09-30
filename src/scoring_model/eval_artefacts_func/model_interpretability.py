import matplotlib.pyplot as plt
import pandas as pd
import shap

from scoring_model.runtime.config import ConfigLoader
from scoring_model.runtime.paths import ProjectPaths
from scoring_model.scripts.predict import MODEL_NAME, model_trained


class ModelInterpreter:

    def __init__(self, config_file: str = "config.yaml") -> None:

        self.config = ConfigLoader().load_yaml(config_file)
        self.paths = ProjectPaths()

        self.random = self.config["rd_seed"]

        self.X_dir = self.paths.processed / "processed_test"
        self.y_dir = self.paths.intermediate / "unprocessed_test"

        self.X_test_processed = pd.read_parquet(self.X_dir / 
                                                 "X_test_processed.parquet"
                                                 )

        self.y_test = pd.read_parquet(self.y_dir / "y_test.parquet")

        self.model_name = MODEL_NAME
        self.model_trained = model_trained

        self.output_dir = (self.paths.reports / self.model_name)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Compute SHAP values and model explainer

    def _shap_values(self):

        """
        Compute SHAP values for the trained model.

        Returns
        -------
        tuple
            SHAP explainer and SHAP explanation object.
        """

        if self.model_name.startswith("logistic_regression"):

            explainer = shap.LinearExplainer(
                                            self.model_trained,
                                            self.X_test_processed[:100]
                                            )

        else:

            explainer = shap.TreeExplainer(self.model_trained)


        shap_values = explainer(self.X_test_processed)

        shap_values.feature_names = self._clean_feature_names(
                                            shap_values.feature_names
                                            )

        return explainer, shap_values


    # Readable feature names (drop ColumnTransformer prefixes)

    @staticmethod
    def _clean_feature_names(feature_names) -> list[str]:

        """
        Remove the "categorical__" / "numerical__" prefixes added by the
        ColumnTransformer so the plots stay readable.
        """

        return [name.split("__", 1)[-1] for name in feature_names]


    # Start every SHAP plot on a fresh figure

    @staticmethod
    def _new_figure() -> None:

        """
        SHAP draws on the current matplotlib figure (plt.gcf()).
        Figures left open by other artefacts (ROC, PR curve, ...) would be
        reused and mixed with the SHAP plot, so close them all first.
        """

        plt.close("all")


    # Get Class risk

    def _risk_class(self, probability: float) -> str:

        """
        Determine the risk class from the probability of default.

        """
        if probability < 0.05:
            return "Low Risk"

        if probability < 0.15:
            return "Medium Risk"

        return "High Risk"

    # Shap summary plot
    def plot_shap_summary(self) -> None:

        """
        Generate global SHAP summary plots.

        Two plots are generated:
        - SHAP beeswarm plot
        - SHAP global importance bar plot
        """

        _, shap_values = self._shap_values()


        #--Beeswarm plot
        self._new_figure()

        shap.summary_plot(shap_values, self.X_test_processed, show=False)

        plt.title(f"SHAP Summary - {self.model_name}")

        plt.tight_layout()

        output = (self.output_dir / f"shap_summary_{self.model_name}.png")

        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()


        #--Variables importance
        self._new_figure()

        shap.summary_plot(shap_values, self.X_test_processed,
                          plot_type="bar", show=False)

        plt.title(f"SHAP Feature Importance - {self.model_name}")

        plt.tight_layout()

        output = (self.output_dir / f"shap_summary_bar_{self.model_name}.png")

        plt.savefig(output, dpi=300, bbox_inches="tight")

        plt.close()


    # Waterfall plot 
    def plot_waterfall(self, client_information: pd.Series | None = None,
                        client_name: str = "ClientX") -> None:

        """
        Generate a local SHAP waterfall explanation.

        Parameters
        ----------
        client_name : str
            Name of the client displayed on the graph.

        observation_index : int
            Row number of the observation in the test dataset.
            Used only when client_information is None.

        client_information : pd.Series | None, default=None
            Client information to explain.

            If None:
                the observation is retrieved from the test set.

            If a pd.Series is provided:
                the provided client information is used directly.
        """


        explainer, shap_values = self._shap_values()

        # Check for input values

        if client_information is None:

            client_input = self.X_test_processed.iloc[[self.random]]

            shap_explanation = shap_values[self.random]

        elif isinstance(client_information, pd.Series):

            client_input = client_information.to_frame().T[
                                            self.X_test_processed.columns
                                            ]

            shap_explanation = explainer(client_input)[0]

            shap_explanation.feature_names = self._clean_feature_names(
                                            client_input.columns
                                            )

        else:

            raise TypeError("client_information must be a pandas Series or None.")


        # Compute probability of default and predict risk class

        probability = self.model_trained.predict_proba(client_input)[0, 1]

        risk_class = self._risk_class(probability)


        #-- Plot waterfall

        self._new_figure()

        shap.plots.waterfall(shap_explanation, max_display=10, show=False)

        fig = plt.gcf()

        fig.suptitle(
                    f"Client: {client_name}  |  "
                    f"Probability of Default: {probability:.2%}  |  "
                    f"Risk Class: {risk_class}",
                    x=0.5,
                    y=1.08,
                    fontsize=12,
                )

        fig.text(
                0.5,
                -0.08,
                "SHAP values in log-odds (model output before sigmoid)",
                ha="center",
                fontsize=9,
                color="grey",
                )

        output = (self.output_dir / f"waterfall_{client_name}.png")

        fig.savefig(output, dpi=300, bbox_inches="tight")

        plt.close(fig)
