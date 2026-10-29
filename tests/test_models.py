from ensemble_benchmark.models import build_models


def test_expected_models_exist_and_fit():
    models = build_models()
    assert set(models) == {"Random Forest", "Gradient Boosting", "XGBoost"}
    # Smoke-test fitting on a small deterministic synthetic dataset.
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=60, n_features=6, random_state=42)
    for model in models.values():
        model.fit(X, y)
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (60, 2)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
