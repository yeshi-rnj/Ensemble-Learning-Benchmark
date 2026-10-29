"""Feature importance extraction and cross-model rank comparison."""

import pandas as pd


def compare_importances(models, feature_names):
    """Return wide importance table with one column per model."""
    tables = []
    for name, model in models.items():
        if not hasattr(model, "feature_importances_"):
            continue
        tables.append(
            pd.DataFrame(
                {
                    "feature": feature_names,
                    name: model.feature_importances_,
                }
            )
        )
    if not tables:
        return pd.DataFrame(columns=["feature"])
    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table, on="feature", how="outer")
    for column in merged.columns[1:]:
        merged[f"{column} rank"] = merged[column].rank(ascending=False, method="min")
    return merged.sort_values(merged.columns[1], ascending=False)
