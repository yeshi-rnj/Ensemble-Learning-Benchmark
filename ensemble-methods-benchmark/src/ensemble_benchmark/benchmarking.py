"""Run the complete benchmark and export reproducible artifacts."""
from pathlib import Path
import json
import platform
import time
import pandas as pd
import sklearn
import xgboost
from .data import load_data
from .models import build_models
from .tuning import tune_models
from .evaluation import evaluate_model
from .importance import compare_importances
from .visualization import create_plots


def run_benchmark(output_root="reports", quick=False, random_state=42):
    """Run baseline/tuned experiments and write reports and figures."""
    root = Path(output_root)
    results_dir, figures_dir = root / "results", root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, features = load_data(random_state=random_state)
    baseline_models, tuned_models, searches = tune_models(
        X_train, y_train, random_state=random_state, quick=quick
    )

    rows, tuned_details = [], {}
    # Baseline measurements use fresh estimators; tuning itself never sees test data.
    for name, baseline in build_models(random_state).items():
        row, _ = evaluate_model(name, baseline, X_train, y_train, X_test, y_test, "baseline")
        rows.append(row)

    for name, model in tuned_models.items():
        row, detail = evaluate_model(name, model, X_train, y_train, X_test, y_test, "tuned")
        rows.append(row)
        tuned_details[name] = detail

    metrics = pd.DataFrame(rows)
    metrics.to_csv(results_dir / "all_metrics.csv", index=False)
    metrics[metrics.variant == "baseline"].to_csv(results_dir / "baseline_metrics.csv", index=False)
    metrics[metrics.variant == "tuned"].to_csv(results_dir / "tuned_metrics.csv", index=False)
    metrics[metrics.variant == "tuned"][["model", "training_seconds", "prediction_seconds"]].to_csv(
        results_dir / "training_times.csv", index=False
    )

    importance = compare_importances(tuned_models, features)
    importance.to_csv(results_dir / "feature_importances.csv", index=False)
    top_sets = {
        name: set(importance.nlargest(10, name)["feature"])
        for name in tuned_models if name in importance.columns
    }
    overlap_rows = []
    for a in top_sets:
        for b in top_sets:
            if a < b:
                overlap_rows.append({
                    "model_a": a, "model_b": b,
                    "shared_top10_count": len(top_sets[a] & top_sets[b]),
                    "shared_features": "; ".join(sorted(top_sets[a] & top_sets[b])),
                })
    pd.DataFrame(overlap_rows).to_csv(results_dir / "top_feature_overlap.csv", index=False)

    search_meta = {
        name: {
            "best_params": search.best_params_,
            "best_cv_roc_auc": float(search.best_score_),
            "candidate_configurations": len(search.cv_results_["params"]),
        }
        for name, search in searches.items()
    }
    metadata = {
        "random_state": random_state, "test_size": 0.2, "cv_folds": 5,
        "quick_mode": quick, "n_train": len(X_train), "n_test": len(X_test),
        "n_features": len(features), "positive_class": "malignant",
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "searches": search_meta,
    }
    (results_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    create_plots(metrics, tuned_details, tuned_models, X_test, y_test, importance, figures_dir)

    winner = metrics[metrics.variant == "tuned"].sort_values("roc_auc", ascending=False).iloc[0]
    report = [
        "# Ensemble Methods — Experiment Report", "",
        "> Generated from the current run. Educational use only; not a diagnostic tool.", "",
        "## Tuned model results", "",
        metrics[metrics.variant == "tuned"].round(4).to_markdown(index=False), "",
        "## Hyperparameter search", "",
        "```json", json.dumps(search_meta, indent=2), "```", "",
        "## Conclusion", "",
        f"On this split, **{winner['model']}** achieved the highest test ROC-AUC "
        f"({winner['roc_auc']:.4f}). This is a dataset- and split-specific observation, "
        "not evidence that the model is universally best.", "",
        "Compare F1 and malignant-class recall alongside ROC-AUC. Inspect the train-test "
        "gap for overfitting and the training-time plot for practical cost. Random Forest "
        "uses bagging to reduce variance; Gradient Boosting and XGBoost build trees "
        "sequentially. Feature-importance agreement is summarized in the exported CSV.", "",
        "## Limitations", "",
        "One dataset and one held-out split do not establish generalization across datasets. "
        "Feature importance is not causal evidence, and this project does not establish "
        "clinical validity.", "",
    ]
    (root / "final_report.md").write_text("\n".join(report), encoding="utf-8")
    return metrics, metadata
