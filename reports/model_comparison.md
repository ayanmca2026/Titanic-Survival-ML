# Model Comparison Report

## 1. Models Evaluated
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier

## 2. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## 3. Performance Summary
| Model | Accuracy | ROC-AUC | Training Time |
|---|---|---|---|
| XGBoost | 0.84 | 0.88 | Fast |
| LightGBM | 0.83 | 0.87 | Very Fast |
| Random Forest | 0.82 | 0.86 | Moderate |

## 4. Best Model Selection
XGBoost was selected for deployment due to the best balance of accuracy and generalization on the validation set.
