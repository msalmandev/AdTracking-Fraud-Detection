
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

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

# Create a list to store results for visualization
results_visualization = []

# Hyperparameter Set 1
param_grid_1 = {'learning_rate': [0.05, 0.1, 0.15], 'n_estimators': [90, 100, 110]}
grid_search_1 = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid_1, 
                             cv=3, n_jobs=-1, scoring='roc_auc', verbose=1)
grid_search_1.fit(X_train_scaled, y_train)
results_visualization.append(('Set 1', grid_search_1.best_score_))

# Hyperparameter Set 2
param_grid_2 = {'max_depth': [2, 3, 4], 'subsample': [0.6, 0.7, 0.8]}
grid_search_2 = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid_2, 
                             cv=3, n_jobs=-1, scoring='roc_auc', verbose=1)
grid_search_2.fit(X_train_scaled, y_train)
results_visualization.append(('Set 2', grid_search_2.best_score_))

# Hyperparameter Set 3
param_grid_3 = {'colsample_bylevel': [0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8]}
grid_search_3 = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid_3, 
                             cv=3, n_jobs=-1, scoring='roc_auc', verbose=1)
grid_search_3.fit(X_train_scaled, y_train)
results_visualization.append(('Set 3', grid_search_3.best_score_))

# Hyperparameter Set 4
param_grid_4 = {'min_child_weight': [1, 2, 3], 'scale_pos_weight': [0.4, 0.5, 0.6]}
grid_search_4 = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid_4, 
                             cv=3, n_jobs=-1, scoring='roc_auc', verbose=1)
grid_search_4.fit(X_train_scaled, y_train)
results_visualization.append(('Set 4', grid_search_4.best_score_))

# Hyperparameter Set 5
param_grid_5 = {'gamma': [0.1, 0.2, 0.3], 'reg_alpha': [0.05, 0.1, 0.15]}
grid_search_5 = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid_5, 
                             cv=3, n_jobs=-1, scoring='roc_auc', verbose=1)
grid_search_5.fit(X_train_scaled, y_train)
results_visualization.append(('Set 5', grid_search_5.best_score_))

# Visualization
names, scores = zip(*results_visualization)

plt.figure(figsize=(12, 6))
plt.barh(names, scores, color='blue')
plt.xlabel('ROC-AUC Score')
plt.title('Hyperparameter Tuning Visualization')
plt.show()
