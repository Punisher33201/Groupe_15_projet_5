import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_shap_plot(input_df, model, scaler, X_train, model_name):
    input_scaled = scaler.transform(input_df)
    if model_name in ["RandomForest", "LGBM", "CatBoost"]:
        explainer = shap.TreeExplainer(model.named_steps['model'])
        shap_values = explainer.shap_values(input_scaled)
        fig, ax = plt.subplots()
        shap.plots.waterfall(explainer.expected_value[1], shap_values[1][0], input_df.columns, ax=ax, show=False)
        return fig
    elif model_name == "SVC":
        def predict_proba_svc(X):
            decision = model.named_steps['model'].decision_function(X)
            proba_pos = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba_pos, proba_pos])
        background = scaler.transform(X_train.sample(50, random_state=42))
        explainer = shap.KernelExplainer(predict_proba_svc, background)
        shap_values = explainer.shap_values(input_scaled, nsamples=100)
        fig, ax = plt.subplots()
        shap.plots.waterfall(explainer.expected_value[1], shap_values[1][0], input_df.columns, ax=ax, show=False)
        return fig
    else:
        return None
    
    