from ensemble_benchmark.data import load_data
from ensemble_benchmark.models import build_models
from ensemble_benchmark.importance import compare_importances


def test_feature_importance_dimensions():
    X_train, _, y_train, _, features = load_data()
    models = build_models()
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    table = compare_importances(fitted, features)
    assert len(table) == len(features)
    for name in fitted:
        assert name in table.columns
        assert table[name].notna().all()
