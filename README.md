# Groupe 15 — Project 5
Hello and welcome! 👋  
We hope you have a great day.

This repository contains the **Coding Week Project** developed by the **Debugging Squad**.

## Project Overview
The goal of this project is to build a **Medical Decision Support Application** for the diagnosis of **pediatric appendicitis** using machine learning techniques.

The system analyzes symptoms and clinical data to help pediatricians make more accurate diagnostic decisions. The project also focuses on **model explainability** using SHAP.

## Team
Debugging Squad

## Coding Week
March 09 – March 15, 2026

## Repository Structure

```
├── .github/workflows
│ └──ci.yml
├── app
│ └──app.py   #Interface deployed
├── models
│ ├── best_model_CatBoostClassifier.joblib
│ ├── best_model_LGBMClassifier.joblib
│ ├── best_model_RandomForestClassifier.joblib
│ └── best_model_SVC.joblib
├── notebook
│ └── coding_week_project_5.ipynb
├── src
│ ├── config.py
│ ├── data_loader.py
│ ├── data_processing.py
│ ├── evaluate_model.py
│ └── train_model.py
├── tests/ # Tests unitaires 
│ └──pytest.py
├──.gitattributes
├──.gitignore
├──pyproject.toml
├──README.md
├──requirements.txt
└──uv.lock

## Work Methodology

To ensure an efficient organization of the project, our team adopted a structured and collaborative workflow. The project was divided into four main components:

- **Data Processing**
- **Model Training**
- **SHAP Analysis**
- **Interface Development**

For each component, one team member was assigned as the main responsible for the task, supported by one or two assistants. This distribution allowed us to manage the workload effectively while maintaining clear responsibilities within the team.

Despite this task division, we emphasized **team collaboration and collective learning** throughout the project. We regularly worked together to discuss the objectives, understand the different steps of the workflow, and solve technical challenges as a group.

This collaborative approach was particularly important during the **data analysis phase**, which was a new area for most of our team members. By sharing knowledge, asking questions, and exploring the dataset together, we were able to better understand the concepts involved and significantly improve our skills in data analysis and machine learning.

Overall, this project was not only an opportunity to develop a technical solution but also a valuable experience in **teamwork, knowledge sharing, and collaborative problem-solving**.

## 🩺 Clinical Decision-Support for Pediatric Appendicitis
Groupe_15_Projet_5

This repository hosts the project of **Group_15_Project_5**. Our objective is to develop a clinical decision-support application designed to assist pediatricians in accurately diagnosing appendicitis in children. By analyzing patient symptoms and clinical test results, the application provides data-driven insights to support medical professionals. Given the critical nature of this medical context, the tool aims to enhance diagnostic precision and optimize patient outcomes 


## Key Features

Key Features :
 Integration of data such as : **Length_of_Stay , Alvarado_Score , Appendix_Diameter, WBC_Count, Neutrophil_Percentage, Segmented_Neutrophils, Body_Temperature, Paedriatic_Appendicitis_Score**


## 🛠️ Tech Stack

* **Language:** Python 3.10
* **Data Science:** Pandas, Scikit-learn, Matplotlib
* **Explainability:** SHAP / LIME
* **Interface:** Streamlit 

🚀 Getting Started

### Prerequisites

Ensure you have Python installed. You can install the required dependencies using:

```bash
pip install -r requirements.txt

```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/votre-compte/Groupe_15_projet_5.git

```


2. Navigate to the project directory:
```bash
cd Groupe_15_projet_5

```


3. Train the model:
```bash
python src/train_model.py
```

4. Run the application:
```bash
streamlit run app/app.py
```

5. Run the tests:
```bash
pytest tests/
```
### Automated Tests
The test suite covers:
- **Data loading** verification (`test_data_loader.py`)
- **Data processing** pipeline validation (`test_data_processing.py`)
- **Model training** verification (`test_train_model.py`)
- **SHAP analysis** integration testing (`test_shap_analysis.py`)
- **Inference** pipeline validation (`test_inference.py`)

Tests are automatically executed via **GitHub Actions** on every push.
```

📊 Methodology


Our model is trained on   the dataset : https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis


⚠️ Medical Disclaimer
**This application is a decision-support tool and is NOT intended to replace professional medical judgment, diagnosis, or treatment. The final clinical decision remains the sole responsibility of the healthcare professional.**

👥 Contributors
Group 15 - [École Centrale Casablanca]

-Mame Lesse FAYE (punisher33201)
 
-Mariam KRISSE (mmmkkk26)

-Sahar BELGHITH

-Marie-Reine SOGNON (sognonmariereine1960)

-Cham Samuel Chedrack BOTI (LeSamCham)

---
## 🔧 Data Preprocessing

### Missing Values
The dataset contains missing values in several numerical columns.
We handled them by imputing with the **median**, which is robust to
the right-skewed distributions and clinical outliers present in the data.

### Outliers
Outliers were detected using the **IQR method** (Q1 - 1.5×IQR, Q3 + 1.5×IQR).
Given the medical context, clinically significant outliers (e.g., extreme CRP
or WBC values) were preserved and handled using **RobustScaler** during
normalization, rather than being removed.

---
### Correlation
A Pearson correlation analysis was performed between all numerical features.
No pair of features showed extreme multicollinearity. Features with |r| ≥ 0.3 
with the target variable were considered relevant predictors:
- **Length_of_Stay (0.66)** → strong correlation with target
- **CRP (0.55)** → moderate-strong correlation with target
- **WBC_Count (0.37)** → moderate correlation with target
- **Alvarado_Score (0.30)** → moderate correlation with target

No features were removed, as tree-based models (Random Forest, CatBoost) 
are robust to correlated features.

## 📊 Dataset Analysis & Class Balance

The dataset presents a **mild class imbalance**, with approximately 60% appendicitis 
cases and 40% non-appendicitis cases.

Rather than applying resampling techniques, we addressed this imbalance 
through two strategies:

- **Stratified splitting:** `train_test_split` with `stratify=y` ensures that both 
  train and test sets preserve the original class distribution.
- **Stratified cross-validation:** `StratifiedKFold(n_splits=5)` was used during 
  hyperparameter tuning to maintain class proportions across all folds.

These choices ensured that the models were evaluated fairly across both classes. 
The impact is reflected in the consistently high Recall scores for both classes, 
particularly for the appendicitis class where missing a true case carries the 
highest clinical risk.
**Conclusion:** Stratified splitting and cross-validation effectively compensated 
for the imbalance, resulting in high and balanced Recall scores across both classes.
No resampling techniques (such as SMOTE or undersampling) were applied, 
as stratified strategies proved sufficient to handle the mild imbalance.

### 🧠 Model Performance & Selection

To ensure the highest diagnostic accuracy, we evaluated multiple machine learning architectures. We prioritize **Sensitivity (Recall)**, as missing an appendicitis diagnosis in a pediatric patient carries a higher clinical risk than a false positive.

#### Evaluated Models

We trained and compared the following algorithms:

| Model | Primary Advantage |
| --- | --- |
| **CatBoost** | Excellent handling of categorical clinical features. |
| **LightGBM (LGBM)** | High performance and speed on tabular data. |
| **Random Forest** | Robustness and clear feature importance ranking. |
| **SVC** | Effective in high-dimensional feature spaces. |

#### Model Evaluation Strategy

We compare these models using a standardized test set to ensure reliability. The evaluation focuses on:

1. **Sensitivity (Recall):** Minimizing "false negatives" to avoid missing acute cases.
2. **Specificity:** Reducing "false positives" to prevent unnecessary medical interventions.
3. **Interpretability:** Using SHAP values to explain the contribution of each symptom to the final prediction.

#### MODEL'S RESULTS

 **CatBoost** 
 Score = 0.9426751592356688
AUC score = 0.9885752688172043

confusion's matrix :
[[90  3]
 [ 6 58]]

classification's report :
                 precision    recall  f1-score   support

   appendicitis       0.94      0.97      0.95        93
no appendicitis       0.95      0.91      0.93        64

       accuracy                           0.94       157
      macro avg       0.94      0.94      0.94       157
   weighted avg       0.94      0.94      0.94       157

![alt text](image.png)

 **LightGBM** 
 Score = 0.9554140127388535
AUC score = 0.9791666666666666

confusion's matrix :
[[91  2]
 [ 5 59]]

classification's report :
                 precision    recall  f1-score   support

   appendicitis       0.95      0.98      0.96        93
no appendicitis       0.97      0.92      0.94        64

       accuracy                           0.96       157
      macro avg       0.96      0.95      0.95       157
   weighted avg       0.96      0.96      0.96       157

![alt text](image-1.png)

 **Random Forest** 
Score = 0.9554140127388535
AUC score = 0.9899193548387097

confusion's matrix :
[[90  3]
 [ 4 60]]

classification's report :
                 precision    recall  f1-score   support

   appendicitis       0.96      0.97      0.96        93
no appendicitis       0.95      0.94      0.94        64

       accuracy                           0.96       157
      macro avg       0.95      0.95      0.95       157
   weighted avg       0.96      0.96      0.96       157

![alt text](image-2.png)

 **SVC(SVM)** 
 Score = 0.8789808917197452
AUC score = 0.9339717741935484

confusion's matrix :
[[88  5]
 [14 50]]

classification's report :
                 precision    recall  f1-score   support

   appendicitis       0.86      0.95      0.90        93
no appendicitis       0.91      0.78      0.84        64

       accuracy                           0.88       157
      macro avg       0.89      0.86      0.87       157
   weighted avg       0.88      0.88      0.88       157

![alt text](image-3.png)

*Note: The primary objective is to maximize Sensitivity to ensure no acute appendicitis cases are overlooked by the system.*

**➡️ Selected Model: Random Forest** — chosen for its highest AUC (0.9899) combined 
with strong Recall (0.97) on the appendicitis class, making it the most reliable 
model for clinical decision support.

## **SHAP Explainability**

## 🔬 Model Explainability & Clinical Insights

To ensure our application is not a "black box," we utilized **SHAP (SHapley Additive exPlanations)** to interpret how our machine learning models arrive at a diagnosis. This is critical in a pediatric context where clinical trust is paramount.

## 🔍 The Clinical Imperative: ##

## Why Explainability Matters (SHAP Analysis) ##

Scenario: The Unjustified Loan Rejection

Imagine yourself as a bank customer using a complex Machine Learning model to decide whether to grant you a mortgage or not.

You enter your banking details and parameters and request a loan. The model generates an automatic response: "Rejected."
The first question that comes to your mind is "Why?"

You have the right to know the reasons for the rejection.
You approach a manager and he replies that these are the results of an algorithm, without being able to tell you the real reasons for the refusal...
This is obviously quite frustrating.

 It is when this quest for understanding the decision factors comes into play that **SHAP analysis** enters the picture.

### Key Findings from SHAP Analysis

![alt text](image-4.png)

![alt text](image-5.png)

Our analysis reveals that the models prioritize established clinical diagnostic protocols:

* **Dominance of Clinical Scores:** The **Paediatric Appendicitis Score (PAS)** and the **Alvarado Score** are the primary drivers of model output. This validates that our machine learning approach aligns with, and reinforces, standard medical diagnostic workflows.
* **Feature Redundancy:** Biological markers (such as `WBC_Count` and `Appendix_Diameter`) show lower individual SHAP values. This suggests that the information provided by these variables is often already captured within the aggregate clinical scores.
* **Model Divergence:** Our comparative analysis between different model architectures (CatBoost, LGBM, RF, SVC) highlights how different algorithms weigh these scores. For example, some models favor the PAS score while others prioritize the Alvarado score, illustrating the importance of choosing a model that minimizes bias toward a specific clinical tool.
**➡️ Top influential features:** Paediatric_Appendicitis_Score, Alvarado_Score, 
WBC_Count, and Appendix_Diameter were the strongest predictors identified by SHAP.

### Clinical Interpretation

While clinical scores are foundational, our model uses these variables to refine the diagnostic threshold. By identifying the specific features that influence each prediction, we ensure that clinicians can understand the logic behind the application's suggestion, thereby facilitating **Shared Decision-Making**.

> **Note on Explainability:** We prioritize models that offer high transparency, ensuring that when the application flags a potential case of appendicitis, the contributing factors—such as specific symptom intensity or lab results—are clearly visible to the attending pediatrician.

## 🧠 Prompt Engineering
Prompt engineering was applied during the **data processing phase** using DeepSeek. 
Structured prompts with explicit context led to domain-relevant recommendations 
(median imputation, RobustScaler). See [PROMPT_ENGINEERING.md](./PROMPT_ENGINEERING.md) 
for full documentation.
---


---

## 📝 Conclusion

This project successfully delivered a **clinical decision-support application** 
for pediatric appendicitis diagnosis, combining robust machine learning with 
full explainability through SHAP analysis.

Key takeaways:
- **Random Forest** emerged as the best model with 95.5% accuracy and AUC of 0.9899.
- **SHAP analysis** confirmed that clinical scores (PAS, Alvarado) are the most 
  influential predictors, aligning with established medical practice.
- **Prompt engineering** with DeepSeek accelerated the data processing phase 
  by producing domain-relevant, accurate code from the first attempt.
- The application provides pediatricians with **transparent, interpretable predictions**, 
  supporting informed clinical decision-making.
Beyond the technical results, this project was a valuable learning experience 
for the entire team. We significantly improved our skills in:
- **GitHub** — version control, branching, CI/CD with GitHub Actions
- **Data Analysis** — EDA, outlier detection, missing value handling, correlation analysis
- **Machine Learning** — model training, performance evaluation

> This project was developed as part of Coding Week — March 09–15, 2026  
> École Centrale Casablanca — Group 15 — Debugging Squad



