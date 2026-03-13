import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.append(".")

from src.inference import InferenceEngine
from src.shap_analysis import get_shap_plot

# Initialize the engine
engine = InferenceEngine()

st.title("Pediatric Appendicitis Prediction App")

st.markdown("""
This app predicts the probability of pediatric appendicitis based on clinical features using trained machine learning models.
Select a model and input the feature values to get a prediction along with SHAP explainability.
""")

# User inputs
st.header("Input Features")
user_input = {}
for col in engine.cols_used:
    min_val = float(engine.X_train[col].min())
    max_val = float(engine.X_train[col].max())
    mean_val = float(engine.X_train[col].mean())
    user_input[col] = st.slider(col, min_val, max_val, mean_val, step=0.1)

# Model selection
model_name = st.selectbox("Select Model", list(engine.models.keys()))

if st.button("Predict"):
    pred_proba = engine.predict(user_input, model_name)
    st.subheader("Prediction")
    st.write(f"Probability of Appendicitis: {pred_proba:.3f}")
    if pred_proba > 0.5:
        st.write("**Prediction: Appendicitis**")
    else:
        st.write("**Prediction: No Appendicitis**")
    
    # SHAP Explainability
    st.subheader("SHAP Explanation")
    input_df = pd.DataFrame([user_input])
    fig = get_shap_plot(input_df, engine.models[model_name], engine.scaler, engine.X_train, model_name)
    if fig:
        st.pyplot(fig)
    else:
        st.write("SHAP explanation not available for this model.")
