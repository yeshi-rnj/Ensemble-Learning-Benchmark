"""Create plots from actual experiment outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay


def create_plots(metrics, details, models, X_test, y_test, importance, output_dir):
    """Generate all core benchmark plots and close figures after saving."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tuned = metrics[metrics["variant"] == "tuned"].copy()

    metric_cols = ["accuracy", "precision_malignant", "recall_malignant", "f1_malignant", "roc_auc"]
    ax = tuned.set_index("model")[metric_cols].plot(kind="bar", figsize=(11, 6))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Tuned model metric comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out / "tuned_metrics.png", dpi=160)
    plt.close()

    ax = metrics.pivot(index="model", columns="variant", values="roc_auc").plot(
        kind="bar", figsize=(8, 5)
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Baseline vs tuned ROC-AUC")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out / "baseline_vs_tuned_auc.png", dpi=160)
    plt.close()

    ax = tuned.set_index("model")["training_seconds"].plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Seconds")
    ax.set_title("Tuned model training time")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out / "training_time.png", dpi=160)
    plt.close()

    ax = tuned.set_index("model")[["train_roc_auc", "roc_auc"]].plot(kind="bar", figsize=(8, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Train vs test ROC-AUC")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out / "train_test_auc.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.set_title("ROC curves — tuned models")
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=160)
    plt.close(fig)

    for name, detail in details.items():
        if name not in models:
            continue
        matrix = detail["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} confusion matrix")
        fig.tight_layout()
        fig.savefig(out / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=160)
        plt.close(fig)

    value_cols = [c for c in importance.columns if c in models]
    if value_cols:
        top_features = (
            importance.assign(_mean=importance[value_cols].mean(axis=1))
            .nlargest(15, "_mean")
            .sort_values("_mean")
        )
        ax = top_features.set_index("feature")[value_cols].plot(kind="barh", figsize=(11, 8))
        ax.set_xlabel("Feature importance")
        ax.set_title("Top 15 feature importances across models")
        plt.tight_layout()
        plt.savefig(out / "feature_importance_top15.png", dpi=160)
        plt.close()
