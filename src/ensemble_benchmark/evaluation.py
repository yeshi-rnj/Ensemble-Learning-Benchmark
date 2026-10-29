"""Consistent classification metrics and confusion matrices."""

import time

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(name, model, X_train, y_train, X_test, y_test, variant):
    """Measure fit/predict time and evaluate malignant=positive metrics."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predictions = model.predict(X_test)
    predict_seconds = time.perf_counter() - start
    probabilities = model.predict_proba(X_test)[:, 1]
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, probabilities)
    row = {
        "model": name,
        "variant": variant,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_malignant": precision_score(y_test, predictions, zero_division=0),
        "recall_malignant": recall_score(y_test, predictions, zero_division=0),
        "f1_malignant": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": test_auc,
        "train_roc_auc": train_auc,
        "train_test_auc_gap": train_auc - test_auc,
        "training_seconds": fit_seconds,
        "prediction_seconds": predict_seconds,
    }
    details = {
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            zero_division=0,
        ),
        "probabilities": probabilities,
        "predictions": predictions,
    }
    return row, details
