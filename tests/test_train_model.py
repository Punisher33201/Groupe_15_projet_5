import joblib

from sklearn.svm import SVC

import src.train_model as train_model


class DummyGridSearchCV:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.best_params_ = {"model__C": 1}
        self.best_score_ = 0.42
        self.best_estimator_ = "best"
        self.fit_called = False

    def fit(self, X, y):
        self.fit_called = True
        self.fit_args = (X, y)
        return self


def test_best_params_invokes_grid_search_and_saves(monkeypatch, tmp_path):
    called = {}

    # Patch GridSearchCV so we don't run a full hyperparameter search.
    monkeypatch.setattr(train_model, "GridSearchCV", lambda *args, **kwargs: DummyGridSearchCV(*args, **kwargs))

    # Patch joblib.dump so we don't write to disk.
    def fake_dump(obj, path, **kwargs):
        called["dumped"] = (obj, str(path))

    monkeypatch.setattr(joblib, "dump", fake_dump)

    # Force model output directory to be under the temporary folder.
    monkeypatch.setattr(train_model, "Path", lambda p: tmp_path / p)

    estimator = train_model.best_params(
        SVC(),
        {"model__C": [0.1, 1.0]},
        X_train_scaled=[[0, 1], [1, 0]],
        y_train=[0, 1],
    )

    assert estimator == "best"
    assert called.get("dumped") is not None
    assert "best_model_SVC" in called["dumped"][1]
