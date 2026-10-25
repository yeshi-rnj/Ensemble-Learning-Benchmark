"""Baseline model definitions."""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


def build_models(random_state: int = 42):
    """Create fresh baseline estimators with reproducible seeds."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1,
            class_weight=None
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            objective="binary:logistic", eval_metric="logloss",
            random_state=random_state, n_jobs=2, tree_method="hist"
        ),
    }


def parameter_grids(quick: bool = False):
    """Small CPU-friendly search spaces for each model."""
    if quick:
        return {
            "Random Forest": {
                "n_estimators": [100],
                "max_depth": [None, 8],
                "min_samples_leaf": [1],
            },
            "Gradient Boosting": {
                "n_estimators": [100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2],
            },
            "XGBoost": {
                "n_estimators": [100],
                "max_depth": [2, 3],
                "learning_rate": [0.05],
            },
        }
    return {
        "Random Forest": {
            "n_estimators": [100, 250],
            "max_depth": [None, 8, 14],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", 0.8],
        },
        "Gradient Boosting": {
            "n_estimators": [75, 150],
            "learning_rate": [0.03, 0.1],
            "max_depth": [1, 2, 3],
            "subsample": [0.8, 1.0],
            "min_samples_leaf": [1, 3],
        },
        "XGBoost": {
            "n_estimators": [75, 150],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [1.0, 3.0],
        },
    }
