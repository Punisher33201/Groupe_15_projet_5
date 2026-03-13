import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_shap_plot(input_df, model, scaler, X_train, model_name):
    input_scaled = scaler.transform(input_df)

    # Build SHAP explanation using the library's explainer API
    assert model_name in ["RandomForest", "LGBM", "CatBoost"], f"Model {model_name} is not supported"
    explainer = shap.TreeExplainer(model.named_steps['model'])
    explanation = explainer(input_scaled)


    # Select the first row (id_to_explain=0) and positive output (output_to_explain=1)
    # Explanation can be multi-output: shape (n_rows, n_features, n_outputs)
    if explanation.values.ndim == 3:
        chosen = explanation[0, :, 1]
    else:
        chosen = explanation[0]

    shap.plots.waterfall(chosen, show=False)
    return plt.gcf()

    