from scoring_model.eval_artefacts_func.classification_report import ClassificationReport  # noqa: I001
from scoring_model.eval_artefacts_func.confusion_matrix import ConfusionMatrix
from scoring_model.eval_artefacts_func.curve_precision_recall import PrecisionRecallCurve
                                                                        
from scoring_model.eval_artefacts_func.eval_report import EvaluationReport
from scoring_model.eval_artefacts_func.roc_curve import RocCurve
from scoring_model.eval_artefacts_func.train_validation import LearningCurve


def orchestrator() -> None:

    ClassificationReport().build()

    ConfusionMatrix().build()

    RocCurve().build()

    PrecisionRecallCurve().build()

    LearningCurve().build()

    EvaluationReport().build()


if __name__ == "__main__":

    orchestrator()