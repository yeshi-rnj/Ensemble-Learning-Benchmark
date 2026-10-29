# Ensemble Methods — Experiment Report

> Generated from the current run. Educational use only; not a diagnostic tool.

## Tuned model results

| model             | variant   |   accuracy |   precision_malignant |   recall_malignant |   f1_malignant |   roc_auc |   train_roc_auc |   train_test_auc_gap |   training_seconds |   prediction_seconds |
|:------------------|:----------|-----------:|----------------------:|-------------------:|---------------:|----------:|----------------:|---------------------:|-------------------:|---------------------:|
| Random Forest     | tuned     |     0.9737 |                     1 |             0.9286 |          0.963 |    0.9929 |               1 |               0.0071 |             0.437  |               0.0756 |
| Gradient Boosting | tuned     |     0.9649 |                     1 |             0.9048 |          0.95  |    0.9914 |               1 |               0.0086 |             1.003  |               0.0031 |
| XGBoost           | tuned     |     0.9737 |                     1 |             0.9286 |          0.963 |    0.9937 |               1 |               0.0063 |             0.1564 |               0.007  |

## Hyperparameter search

```json
{
  "Random Forest": {
    "best_params": {
      "max_depth": null,
      "max_features": "sqrt",
      "min_samples_leaf": 1,
      "min_samples_split": 2,
      "n_estimators": 100
    },
    "best_cv_roc_auc": 0.9889060887512902,
    "candidate_configurations": 48
  },
  "Gradient Boosting": {
    "best_params": {
      "learning_rate": 0.1,
      "max_depth": 2,
      "min_samples_leaf": 3,
      "n_estimators": 150,
      "subsample": 1.0
    },
    "best_cv_roc_auc": 0.9929824561403509,
    "candidate_configurations": 48
  },
  "XGBoost": {
    "best_params": {
      "colsample_bytree": 1.0,
      "learning_rate": 0.1,
      "max_depth": 2,
      "n_estimators": 150,
      "reg_lambda": 1.0,
      "subsample": 0.8
    },
    "best_cv_roc_auc": 0.9948400412796697,
    "candidate_configurations": 96
  }
}
```

## Conclusion

On this split, **XGBoost** achieved the highest test ROC-AUC (0.9937). This is a dataset- and split-specific observation, not evidence that the model is universally best.

Compare F1 and malignant-class recall alongside ROC-AUC. Inspect the train-test gap for overfitting and the training-time plot for practical cost. Random Forest uses bagging to reduce variance; Gradient Boosting and XGBoost build trees sequentially. Feature-importance agreement is summarized in the exported CSV.

## Limitations

One dataset and one held-out split do not establish generalization across datasets. Feature importance is not causal evidence, and this project does not establish clinical validity.
