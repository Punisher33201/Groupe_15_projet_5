import pandas as pd
from types import SimpleNamespace

import data_loader


def test_load_data_returns_dataframes(monkeypatch):
    """load_data should return X as a DataFrame and y as a Series."""

    X_expected = pd.DataFrame({"a": [1, 2, 3]})
    y_expected = pd.Series([0, 1, 0], name="target")

    fake_dataset = SimpleNamespace(data=SimpleNamespace(features=X_expected, targets=y_expected))

    monkeypatch.setattr(data_loader, "fetch_ucirepo", lambda id: fake_dataset)

    X, y = data_loader.load_data()

    pd.testing.assert_frame_equal(X, X_expected)
    pd.testing.assert_series_equal(y, y_expected)
