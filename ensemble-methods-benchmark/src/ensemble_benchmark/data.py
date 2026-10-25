"""Dataset loading and reproducible train/test splitting."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data(test_size: float = 0.2, random_state: int = 42):
    """Return X_train, X_test, y_train, y_test, feature_names.

    Target is remapped to malignant=1 and benign=0.
    """
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    # Original sklearn target: 0=malignant, 1=benign. Explicitly remap.
    y = (dataset.target == 0).astype(int)
    y.name = "malignant"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, list(X.columns)
