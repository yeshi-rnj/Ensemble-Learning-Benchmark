from ensemble_benchmark.data import load_data
from ensemble_benchmark.evaluation import evaluate_model
from ensemble_benchmark.models import build_models


def test_metrics_are_valid():
    X_train, X_test, y_train, y_test, _ = load_data()
    model = build_models()["Random Forest"]
    row, detail = evaluate_model("Random Forest", model, X_train, y_train, X_test, y_test, "test")
    for metric in [
        "accuracy",
        "precision_malignant",
        "recall_malignant",
        "f1_malignant",
        "roc_auc",
    ]:
        assert 0 <= row[metric] <= 1
    assert detail["confusion_matrix"].shape == (2, 2)
