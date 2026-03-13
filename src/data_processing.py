# importing the necessary libraries
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
from sklearn.metrics import classification_report, confusion_matrix

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

def data_prossess(X, y, cols_used : list):
    data = X.loc[:, cols_used] # subsets of datasets to train futurs models
    for col in cols_used:
        data[col] = data[col].fillna(data[col].median()) # replacing all missing values
    return data

def data_divison(data, y, target : str):
    # data divisions 
    y = y.copy()
    y[target] = y[target].fillna(y[target].mode()[0])
    X_train, X_test, y_train, y_test = train_test_split(data, y[target], test_size = .2, stratify=y[target], random_state = 42)

    scaler = RobustScaler() # reduce or annihilate the negative affect of outliers
    # scal the datas for better exploitation
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


