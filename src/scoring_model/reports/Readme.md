# Model Evaluation Report

## Model

- Model: logistic_regression_v001

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| False | 0.9165 | 0.7403 | 0.8191 | 801 |
| True | 0.1938 | 0.4808 | 0.2762 | 104 |


| Aggregate | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| macro avg | 0.5552 | 0.6105 | 0.5477 | 905 |
| weighted avg | 0.8335 | 0.7105 | 0.7567 | 905 |


## Accuracy

- Accuracy: 0.7105

## Confusion Matrix

![Confusion Matrix](figures/logistic_regression_v001/confusion_matrix_logistic_regression_v001.png)

## ROC Curve

![ROC Curve](figures/logistic_regression_v001/roc_curve_logistic_regression_v001.png)

## Precision-Recall Curve

![Precision-Recall Curve](figures/logistic_regression_v001/precision_recall_curve_logistic_regression_v001.png)

## Learning Curve

![Learning Curve](figures/logistic_regression_v001/learning_curve_logistic_regression_v001.png)

## Model Interpretability (SHAP)

### SHAP Summary

![SHAP Summary](logistic_regression_v001/shap_summary_logistic_regression_v001.png)

### SHAP Feature Importance

![SHAP Feature Importance](logistic_regression_v001/shap_summary_bar_logistic_regression_v001.png)

### SHAP Waterfall - ClientX

![SHAP Waterfall](logistic_regression_v001/waterfall_ClientX.png)
