import sys 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import shap
from sklearn.metrics import make_scorer, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_loader import load_data
from config import param_grid_svm, param_grid_rf, param_grid_lgb, param_grid_cat
from data_processing import model_score, data_prossess, data_divison
sys.path.append('.')

from src.data_processing import load_data

import warnings
warnings.filterwarnings("ignore")


df = load_data()
print(df)


def best_params(model, param_grid : dict, X_train_scaled, y_train, kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), f1_scorer = make_scorer(f1_score, pos_label='appendicitis')):
    # Pipeline avec imputation (si nécessaire) + scaling + mpdel
    pipeline = Pipeline([
        ('model', model)
    ])

    # Grille d'hyperparamètres (les noms sont préfixés par 'mpdel__')
    grid = GridSearchCV(pipeline, param_grid, cv=kf, scoring=f1_scorer, verbose=1)
    grid.fit(X_train_scaled, y_train)

    # Résultats
    print("Meilleurs paramètres :", grid.best_params_)
    print("Meilleur score F1 (validation croisée) :", grid.best_score_)
    return


cols_used = ['Length_of_Stay', 'Alvarado_Score', 'Appendix_Diameter', 'WBC_Count', 'Neutrophil_Percentage', 'Segmented_Neutrophils', 'Body_Temperature', 'Paedriatic_Appendicitis_Score'] # columns we'll used for the rest

X,y = load_data()
data = data_prossess(X, y, cols_used)
X_train_scaled, X_test_scaled, y_train, y_test = data_divison(data, y, 'Diagnosis')



best_params(SVC(), param_grid_svm, X_train_scaled, y_train)
