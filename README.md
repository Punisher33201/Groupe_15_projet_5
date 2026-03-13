# Groupe_15_projet_5

This repository is used for our **Groupe_15_projet_5** wich goal is to develop a clinical decision-support application aimed at assisting pediatricians in the accurate diagnosis of appendicitis in children, using symptoms and clinical test results. Given the critical medical context

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
├──README.md
└──requirements.txt
```



