import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Charger données
data = pd.read_csv("dataset.csv")

# Charger modèle
model = pickle.load(open("lgbm_model.pkl","rb"))

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Dataset Overview","Model Info","Patient Prediction","Model Explanation"]
)

# -------------------------
# PAGE 1 DATASET
# -------------------------

if page == "Dataset Overview":

    st.title("Pediatric Appendicitis Dataset")

    st.write("Dataset preview")
    st.dataframe(data.head())

    st.subheader("Dataset statistics")
    st.write(data.describe())

    st.subheader("Correlation matrix")

    fig, ax = plt.subplots()

    corr = data.corr()

    cax = ax.matshow(corr)

    fig.colorbar(cax)

    st.pyplot(fig)

# -------------------------
# PAGE 2 MODEL
# -------------------------

elif page == "Model Info":

    st.title("Machine Learning Model")

    st.write("Model used: LightGBM")

    st.write("""
    LightGBM is a gradient boosting algorithm optimized for
    performance and efficiency in tabular datasets.
    """)

    st.subheader("Features used")

    features = [
        "Body temperature",
        "Pediatric appendicitis score",
        "Length of stay",
        "Alvarado score",
        "Appendix diameter",
        "WBC count",
        "Neutrophil %",
        "Segmented neutrophil"
    ]

    st.write(features)

# -------------------------
# PAGE 3 PREDICTION
# -------------------------

elif page == "Patient Prediction":

    st.title("Appendicitis Prediction")

    body_temp = st.number_input("Body Temperature")

    pas_score = st.number_input("Pediatric Appendicitis Score")

    length_stay = st.number_input("Length of Stay")

    alvarado = st.number_input("Alvarado Score")

    appendix_diameter = st.number_input("Appendix Diameter")

    wbc = st.number_input("WBC Count")

    neutrophil_percent = st.number_input("Neutrophil %")

    segmented_neutrophil = st.number_input("Segmented Neutrophil")

    features = np.array([[
        body_temp,
        pas_score,
        length_stay,
        alvarado,
        appendix_diameter,
        wbc,
        neutrophil_percent,
        segmented_neutrophil
    ]])

    if st.button("Predict"):

        prediction = model.predict(features)

        proba = model.predict_proba(features)

        if prediction[0] == 1:

            st.error("High probability of appendicitis")

        else:

            st.success("Low probability of appendicitis")

        st.write("Probability:",proba[0][1])

# -------------------------
# PAGE 4 SHAP
# -------------------------

elif page == "Model Explanation":

    st.title("Model Explanation with SHAP")

    sample = data.iloc[:50]

    X = sample[[
        "Body temperature",
        "Pediatric appendicitis score",
        "Length of stay",
        "Alvarado score",
        "Appendix diameter",
        "WBC count",
        "Neutrophil %",
        "Segmented neutrophil"
    ]]

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    fig = plt.figure()

    shap.summary_plot(shap_values,X,show=False)

    st.pyplot(fig)