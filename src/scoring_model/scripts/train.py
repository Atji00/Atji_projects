from scoring_model.dataset.loading import RawDataLoader
from scoring_model.dataset.splitting import TrainTestSplitter
from scoring_model.dataset.validation import DataValidator
from scoring_model.features.engineering import FeatureEngineer
from scoring_model.features.preprocessing import Processor
from scoring_model.models.models_loader import Model
from scoring_model.utils.func_casting import casting
from scoring_model.utils.func_missings_imputing import impute_missings


def train_orchestration(model_name: str = 'logistic_regression', version: str = 'v001') -> None:

    def stage(step_name: str, step: int, total_step: int = 8) -> None:

        print(f"\nStep[{step}/{total_step}]: {step_name}")


    def stage_status(arg: str | None = None) -> None:

        print("✓ terminated")
        print("-" * 60)

    print("=" * 60)
    print("CREDIT SCORING — TRAINING PIPELINE")
    print("=" * 60)


    # 1. Data Loading 
    stage('Raw Data Loading', 1)
    rawdata = RawDataLoader().load("bank.csv")
    stage_status()

    # 2. Data Scheme Validation
    stage('Data Scheme Validation', 2)
    DataValidator().validate(rawdata)
    stage_status()

    # 3. Columns Casting
    stage('Columns Casting', 3)
    data = rawdata.apply(casting, axis=0)
    stage_status()

    # 4. Features Engineering
    stage('Features Engineering', 4)
    data = FeatureEngineer(data).compute()
    stage_status()

    # 5. Splitting in Train/Test
    stage('Splitting in Train/Test', 5)
    X_train, _, y_train, _ = TrainTestSplitter().split(data)
    stage_status()

    # 6. Imputing NA/None in X_train
    stage('Imputing NA/None in X_train', 6)
    X_train = impute_missings(X_train)
    stage_status()

    # 7. Preproccesing: Columns transformers
    stage('Preproccesing: Columns transformers', 7)
    X_train_processed = Processor(X_train).fit_transform()
    stage_status()

    # 8. Model Training and Saving
    stage('Model Training and Saving', 8)
    Model().train(model_name, version, X_train_processed, y_train)
    stage_status("\n")

    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    
if __name__ == "__main__":

    train_orchestration('random_forest', 'v001')
