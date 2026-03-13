import numpy as np
import pandas as pd

from src.data_processing import data_divison, data_prossess


def test_data_prossess_fills_missing_values_and_subsets_columns():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10, 20, 30]})
    y = pd.Series([0, 1, 0], name="target")

    out = data_prossess(X.copy(), y, cols_used=["a"])

    assert list(out.columns) == ["a"]
    assert out["a"].isna().sum() == 0
    # median of [1.0, 3.0] is 2.0
    assert out.loc[1, "a"] == 2.0


def test_data_divison_splits_and_scales(tmp_path):
    # Create a small balanced dataset with a missing label
    data = pd.DataFrame({"feature": list(range(10))})
    y = pd.DataFrame({"Diagnosis": ["a", "a", "b", "b", "a", "b", "a", "b", None, "a"]})

    X_train, X_test, y_train, y_test = data_divison(data, y, target="Diagnosis")

    # Should have non-empty splits and no NaNs after scaling
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()

    # Target must have expected labels and no NaNs
    assert set(y_train.unique()) <= {"a", "b"}
    assert set(y_test.unique()) <= {"a", "b"}
