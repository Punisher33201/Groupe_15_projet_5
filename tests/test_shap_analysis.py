import pytest
import matplotlib.pyplot as plt
import pandas as pd
from src.shap_analysis import get_shap_plot
from src.inference import InferenceEngine

def test_get_shap_plot_tree_models():
    engine = InferenceEngine()
    input_df = pd.DataFrame([{col: 1.0 for col in engine.cols_used}])
    for model_name in ["RandomForest", "LGBM", "CatBoost"]:
        fig = get_shap_plot(input_df, engine.models[model_name], engine.scaler, engine.X_train, model_name)
        assert fig is not None
        assert isinstance(fig, plt.Figure)

def test_get_shap_plot_svc():
    engine = InferenceEngine()
    input_df = pd.DataFrame([{col: 1.0 for col in engine.cols_used}])
    fig = get_shap_plot(input_df, engine.models["SVC"], engine.scaler, engine.X_train, "SVC")
    assert fig is not None
    assert isinstance(fig, plt.Figure)

def test_get_shap_plot_invalid_model():
    engine = InferenceEngine()
    input_df = pd.DataFrame([{col: 1.0 for col in engine.cols_used}])
    fig = get_shap_plot(input_df, None, engine.scaler, engine.X_train, "Invalid")
    assert fig is None
