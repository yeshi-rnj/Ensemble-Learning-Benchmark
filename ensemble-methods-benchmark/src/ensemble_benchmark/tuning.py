"""Training-only cross-validated hyperparameter search."""
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from .models import build_models, parameter_grids


def tune_models(X_train, y_train, random_state: int = 42, quick: bool = False):
    """Return best estimators, search summaries, and fitted baseline estimators."""
    baselines = build_models(random_state)
    grids = parameter_grids(quick)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    best_models, searches = {}, {}
    for name, model in baselines.items():
        search = GridSearchCV(
            estimator=model,
            param_grid=grids[name],
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_
        searches[name] = search
    return baselines, best_models, searches
