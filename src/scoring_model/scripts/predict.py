import joblib
import pandas as pd

from scoring_model.runtime.paths import ProjectPaths

paths = ProjectPaths()

MODEL_NAME = 'random_forest_v001'

model_trained = joblib.load(paths.models_registry / f"{MODEL_NAME}.joblib")


def predict() -> tuple[pd.DataFrame, pd.Series, pd.Series]:

    #-------------------- Setting directory and loading processor-------------------#

    test_data_dir = paths.intermediate / "unprocessed_test"

    out_dir = (paths.processed/ "processed_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = joblib.load(paths.features / 'processor_fitted.joblib')


    #--------------------------- Loading tests data ------------------------------# 

    X_test = pd.read_parquet(test_data_dir / 'X_test_unprocessed.parquet')

    y_test = pd.read_parquet(test_data_dir / 'y_test.parquet')

    X_test_processed = processor.transform(X_test)

    X_test_processed = pd.DataFrame(
                                        X_test_processed,
                                        columns=processor.get_feature_names_out(),
                                        index=X_test.index,
                                    )

    #-----------------------Predictions, probabilities, saving to .parquet---------#

    y_predicted = pd.Series(
                                model_trained.predict(X_test_processed),
                                index=y_test.index,
                                name="y_pred"
                            )

    y_probability = pd.Series(
                                model_trained.predict_proba(X_test_processed)[:, 1],
                                index=y_test.index,
                                name="y_prob",
                            )
    
    X_test_processed.to_parquet(out_dir / "X_test_processed.parquet", index = False)

    y_predicted.to_frame().to_parquet(out_dir / "y_predicted.parquet", index=False)

    y_probability.to_frame().to_parquet(out_dir / "y_probability.parquet", index=False)

    return X_test_processed, y_predicted, y_probability

if __name__ == '__main__':

    predict()