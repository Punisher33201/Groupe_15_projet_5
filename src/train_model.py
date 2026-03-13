import warnings
from pathlib import Path

from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append(str(Path(__file__).parent))

from src.config import param_grid_svm, param_grid_rf, param_grid_lgb, param_grid_cat
from src.data_loader import load_data
from src.data_processing import data_prossess, data_divison

warnings.filterwarnings("ignore")


def best_params(
    model,
    param_grid: dict,
    X_train_scaled,
    y_train,
    kf: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    f1_scorer=make_scorer(f1_score, pos_label="appendicitis"),
):
    """Optimize hyperparameters and save the best estimator.

    Args:
        model: A scikit-learn estimator instance.
        param_grid: Grid of hyperparameters to search.
        X_train_scaled: Training features (scaled).
        y_train: Training labels.
        kf: Cross-validation splitter.
        f1_scorer: Scoring function.

    Returns:
        The best estimator found by GridSearchCV.
    """

    pipeline = Pipeline([("model", model)])

    grid = GridSearchCV(pipeline, param_grid, cv=kf, scoring=f1_scorer, verbose=1)
    grid.fit(X_train_scaled, y_train)

    print("Meilleurs paramètres :", grid.best_params_)
    print("Meilleur score F1 (validation croisée) :", grid.best_score_)

    import joblib

    model_name = model.__class__.__name__
    model_path = Path(f"models/best_model_{model_name}.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid.best_estimator_, model_path)

    return grid.best_estimator_


def main():
    # Columns selected for training
    cols_used = [
        "Length_of_Stay",
        "Alvarado_Score",
        "Appendix_Diameter",
        "WBC_Count",
        "Neutrophil_Percentage",
        "Segmented_Neutrophils",
        "Body_Temperature",
        "Paedriatic_Appendicitis_Score",
    ]

    X, y = load_data()
    data = data_prossess(X, y, cols_used)
    X_train_scaled, X_test_scaled, y_train, y_test = data_divison(data, y, "Diagnosis")

    best_params(SVC(), param_grid_svm, X_train_scaled, y_train)
    best_params(
        LGBMClassifier(random_state=42, verbose=-1),
        param_grid_lgb,
        X_train_scaled,
        y_train,
    )
    best_params(
        CatBoostClassifier(random_seed=42, verbose=0, allow_writing_files=False),
        param_grid_cat,
        X_train_scaled,
        y_train,
    )
    best_params(RandomForestClassifier(), param_grid_rf, X_train_scaled, y_train)


if __name__ == "__main__":
    main()
