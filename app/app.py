import streamlit as st
import pandas as pd
import numpy as np

import sys
sys.path.append(".")

from src.inference import InferenceEngine
from src.shap_analysis import get_shap_plot

# --------------------------------------------------------------------------------
# App configuration
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Pediatric Appendicitis Decision Support",
    page_icon="🩺",
    layout="wide",
)

# --------------------------------------------------------------------------------
# Engine initialization
# --------------------------------------------------------------------------------
engine = InferenceEngine()

MODEL_HELP = {
    "RandomForest": "Robust ensemble model with solid overall performance.",
    "LGBM": "Fast, accurate gradient boosting model optimized for tabular data.",
    "CatBoost": "Handles numerical and categorical data with strong real-world performance.",
}

# --------------------------------------------------------------------------------
# Sidebar (inputs + context)
# --------------------------------------------------------------------------------
with st.sidebar:
    st.title("🚑 Appendicitis Predictor")
    st.markdown(
        """
        Use the sliders below to provide clinical measurements. Choose a model and click **Predict** to see the probability of pediatric appendicitis.
        
        ✅ Built with results from validated models trained on the UCI pediatric appendicitis dataset.
        """
    )

    st.markdown("---")

    st.header("1) Select a model")
    model_name = st.selectbox("Choose a trained model", list(engine.models.keys()))
    st.caption(MODEL_HELP.get(model_name, ""))

    st.markdown("---")

    st.header("2) Enter clinical features")

    def _make_input_widget(col_name: str, min_val: float, max_val: float, default: float):
        # Choose best widget type based on value range and precision
        if float(min_val).is_integer() and float(max_val).is_integer():
            return st.number_input(
                col_name,
                min_value=int(min_val),
                max_value=int(max_val),
                value=int(round(default)),
                step=1,
            )

        # Use a slider for continuous values
        step = max((max_val - min_val) / 80, 0.01)
        return st.slider(col_name, min_val, max_val, default, step=step)

    user_input: dict[str, float] = {}
    for col in engine.cols_used:
        min_val = float(engine.X_train[col].min())
        max_val = float(engine.X_train[col].max())
        mean_val = float(engine.X_train[col].mean())
        user_input[col] = _make_input_widget(col, min_val, max_val, mean_val)

    st.markdown("---")
    st.header("3) Run prediction")
    predict_button = st.button("Predict")

    with st.expander("Why this app?"):
        st.markdown(
            """
            This tool is intended to support clinicians by providing a data-driven probability score for pediatric appendicitis.

            ⚠️ **Medical Disclaimer:** This is NOT a diagnostic tool and should not replace clinical judgment. Always consult a medical professional.
            """
        )

    with st.expander("Feature definitions"):
        st.markdown(
            """
            - **Length_of_Stay:** Length of hospital stay (days).
            - **Alvarado_Score:** Clinical scoring system for appendicitis symptoms.
            - **Appendix_Diameter:** Measured diameter of the appendix (mm).
            - **WBC_Count:** White blood cell count (cells/µL).
            - **Neutrophil_Percentage:** Percentage of neutrophils in blood.
            - **Segmented_Neutrophils:** Absolute segmented neutrophil count.
            - **Body_Temperature:** Patient temperature (°C).
            - **Paedriatic_Appendicitis_Score:** Pediatric appendicitis clinical score.
            """
        )

# --------------------------------------------------------------------------------
# Main area (results)
# --------------------------------------------------------------------------------
st.title("🧠 Prediction Results")

if not predict_button:
    st.info("Adjust the inputs in the sidebar and click **Predict** to see the results.")
else:
    pred_proba = engine.predict(user_input, model_name)
    is_appendicitis = pred_proba > 0.3

    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.subheader("Prediction")
        st.metric(
            label="Probability of Appendicitis",
            value=f"{pred_proba:.1%}",
            delta="High" if pred_proba > 0.75 else "Medium" if pred_proba > 0.3 else "Low",
        )
        st.markdown(
            "<span style='font-size:1.1rem'>" + (
                "✅ **Appendicitis likely**" if is_appendicitis else "✅ **Appendicitis unlikely**"
            ) + "</span>",
            unsafe_allow_html=True,
        )

    with col2:
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")

    with col3:
        st.subheader("Input summary")
        st.table(pd.DataFrame([user_input]).T.rename(columns={0: "Value"}))

    # SHAP Explainability
    with st.expander("View SHAP explanation (feature impact)", expanded=True):
        input_df = pd.DataFrame([user_input])
        fig = get_shap_plot(input_df, engine.models[model_name], engine.scaler, engine.X_train, model_name)
        if fig:
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("SHAP explanation not available for this model.")

    st.markdown("---")

    with st.expander("Model performance & notes", expanded=False):
        st.markdown(
            """
            These models were trained and evaluated on a pediatric appendicitis dataset. The goal is to maximize **sensitivity**
            (minimize missed cases) while maintaining good **specificity**.

            **Note:** Predictions are based on a pre-trained model and should be treated as supportive information only.
            """
        )
