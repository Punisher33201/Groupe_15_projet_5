param_grid_svm = {
    'model__C': [1, 10, 100],
    'model__gamma': [0.1, 0.01, 0.001],
    'model__kernel': ['rbf']
}

param_grid_rf = {
    'model__n_estimators': [100, 200, 300],        # Nombre d'arbres
    'model__max_depth': [10, 20, 30],        # Profondeur max 
    'model__min_samples_split': [2, 5, 10],        # Échantillons min pour diviser un nœud
    'model__min_samples_leaf': [1, 2, 4],          # Échantillons min dans une feuille
    'model__max_features': ['sqrt', 'log2'],        # Nombre de features à considérer
}

param_grid_lgb = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1],
    'model__num_leaves': [15, 31],
    'model__subsample': [0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.8, 0.9, 1.0],
    'model__scale_pos_weight': [1, 3, 5],      # 3 valeurs (important pour toi)
}

param_grid_cat = {
    'model__iterations': [200, 300],       
    'model__depth': [4, 6],                 
    'model__learning_rate': [0.05, 0.1],    
    'model__scale_pos_weight': [1, 3, 5],    
    'model__l2_leaf_reg': [3]                
}