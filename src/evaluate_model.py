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


def model_score(model, X_train_scaled,X_test_scaled, y_train, y_test, title):
    model.fit(X_train_scaled, y_train) # training the model 

    y_pred = model.predict(X_test_scaled) # making predictions

    # print out the scores
    y_pred_probs = model.predict_proba(X_test_scaled)[:, 1]
    print('Score =', model.score(X_test_scaled, y_test))
    print('AUC score =', roc_auc_score(y_test, y_pred_probs))
    print()

    print("confusion's matrix :")
    print(confusion_matrix(y_test, y_pred))
    print()

    print("classification's report :")
    print(classification_report(y_test, y_pred))
    print()
    print()

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs, pos_label='appendicitis')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.show()
    return

def model_score(model, X_train_scaled,X_test_scaled, y_train, y_test, title):
    model.fit(X_train_scaled, y_train) # training the model 

    y_pred = model.predict(X_test_scaled) # making predictions

    # print out the scores
    y_pred_probs = model.predict_proba(X_test_scaled)[:, 1]
    print('Score =', model.score(X_test_scaled, y_test))
    print('AUC score =', roc_auc_score(y_test, y_pred_probs))
    print()

    print("confusion's matrix :")
    print(confusion_matrix(y_test, y_pred))
    print()

    print("classification's report :")
    print(classification_report(y_test, y_pred))
    print()
    print()

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs, pos_label='appendicitis')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.show()
    return