
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

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

# Create a list to store results for visualization and table
results_comparison = []

# XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train_scaled, y_train)
xgb_score = roc_auc_score(y_test, xgb_model.predict_proba(X_test_scaled)[:, 1])
results_comparison.append(('XGBoost', xgb_score))

# CatBoost
cat_model = CatBoostClassifier(verbose=0)
cat_model.fit(X_train_scaled, y_train)
cat_score = roc_auc_score(y_test, cat_model.predict_proba(X_test_scaled)[:, 1])
results_comparison.append(('CatBoost', cat_score))

# RandomForest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)
rf_score = roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
results_comparison.append(('RandomForest', rf_score))

# Create DataFrame for table
results_df = pd.DataFrame(results_comparison, columns=['Algorithm', 'ROC-AUC Score'])
print(results_df)

# Visualization
names, scores = zip(*results_comparison)

plt.figure(figsize=(12, 6))
plt.barh(names, scores, color='green')
plt.xlabel('ROC-AUC Score')
plt.title('Algorithm Comparison')
plt.show()
