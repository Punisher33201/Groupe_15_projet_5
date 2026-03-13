import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.data_processing import data_prossess

class InferenceEngine:
    def __init__(self):
        # Load and process data
        X, y = load_data()
        self.cols_used = [
            "Length_of_Stay",
            "Alvarado_Score",
            "Appendix_Diameter",
            "WBC_Count",
            "Neutrophil_Percentage",
            "Segmented_Neutrophils",
            "Body_Temperature",
            "Paedriatic_Appendicitis_Score",
        ]
        data = data_prossess(X, y, self.cols_used)
        y_filled = y["Diagnosis"].fillna(y["Diagnosis"].mode()[0])
        X_train, X_test, y_train, y_test = train_test_split(data, y_filled, test_size=0.2, stratify=y_filled, random_state=42)
        self.scaler = RobustScaler()
        self.scaler.fit(X_train)
        self.X_train = X_train  # For SHAP background

        # Load models
        self.models = {
            "SVC": joblib.load("models/best_model_SVC.joblib"),
            "RandomForest": joblib.load("models/best_model_RandomForestClassifier.joblib"),
            "LGBM": joblib.load("models/best_model_LGBMClassifier.joblib"),
            "CatBoost": joblib.load("models/best_model_CatBoostClassifier.joblib"),
        }

    def predict(self, input_dict, model_name):
        input_df = pd.DataFrame([input_dict])
        input_scaled = self.scaler.transform(input_df)
        model = self.models[model_name]
        if model_name == "SVC":
            decision = model.decision_function(input_scaled)[0]
            pred_proba = 1 / (1 + np.exp(-decision))
        else:
            pred_proba = model.predict_proba(input_scaled)[0][1]
        return pred_proba
