
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np

# Load the data
data = pd.read_csv('your_data_file.csv')

# Drop unnecessary columns
X = data.drop(['is_attributed', 'click_time', 'record'], axis=1)
y = data['is_attributed']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Nested GridSearch with best parameters from each set
param_grid_nested = {
    'learning_rate': [0.1],
    'n_estimators': [100],
    'max_depth': [3],
    'subsample': [0.7],
    'colsample_bytree': [0.7],
    'colsample_bylevel': [0.8],
    'min_child_weight': [2],
    'scale_pos_weight': [0.5],
    'gamma': [0.2],
    'reg_alpha': [0.1]
}

grid_search_nested = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid_nested, 
                                  cv=3, n_jobs=-1, scoring='roc_auc', verbose=1)
grid_search_nested.fit(X_train_scaled, y_train)

print(f"Best parameters for Nested GridSearch: {grid_search_nested.best_params_}")
print(f"Best score for Nested GridSearch: {grid_search_nested.best_score_}")

# Fine-Grained Search around the best parameters
param_grid_fine = {
    'learning_rate': [0.05, 0.1, 0.15],
    'n_estimators': [90, 100, 110],
    'max_depth': [2, 3, 4]
}

grid_search_fine = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid_fine, 
                                cv=3, n_jobs=-1, scoring='roc_auc', verbose=1)
grid_search_fine.fit(X_train_scaled, y_train)

print(f"Best parameters for Fine-Grained Search: {grid_search_fine.best_params_}")
print(f"Best score for Fine-Grained Search: {grid_search_fine.best_score_}")

# Different Scoring Metric (F1 Score)
grid_search_f1 = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid_fine, 
                              cv=3, n_jobs=-1, scoring='f1', verbose=1)
grid_search_f1.fit(X_train_scaled, y_train)

print(f"Best parameters for F1 Score: {grid_search_f1.best_params_}")
print(f"Best F1 Score: {grid_search_f1.best_score_}")

# Comparison of Results
results = [grid_search_nested.best_score_, grid_search_fine.best_score_, grid_search_f1.best_score_]
names = ['Nested GridSearch', 'Fine-Grained Search', 'F1 Score']

plt.figure(figsize=(12, 6))
plt.barh(names, results, color='blue')
plt.xlabel('ROC-AUC Score')
plt.title('Comparison of Different Tuning Methods')
plt.show()
