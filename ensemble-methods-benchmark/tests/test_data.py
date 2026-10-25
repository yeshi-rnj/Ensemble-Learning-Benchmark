from ensemble_benchmark.data import load_data


def test_data_dimensions_and_split():
    X_train, X_test, y_train, y_test, features = load_data()
    assert len(features) == 30
    assert X_train.shape[1] == X_test.shape[1] == 30
    assert len(X_train) + len(X_test) == 569
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_split_is_reproducible():
    a = load_data(random_state=7)
    b = load_data(random_state=7)
    assert a[0].equals(b[0])
    assert a[1].equals(b[1])
