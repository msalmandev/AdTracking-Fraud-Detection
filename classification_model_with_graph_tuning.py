
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler

# Load the data
data = pd.read_csv('your_data_file.csv')

# Drop the unnecessary columns
X = data.drop(['is_attributed', 'click_time', 'record'], axis=1)
y = data['is_attributed']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the models
models = [('LR', LogisticRegression()), 
          ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('SVM', SVC()),
          ('XGB', XGBClassifier()),
          ('RF', RandomForestClassifier())]

# Evaluate each model using k-fold cross-validation
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='roc_auc')
    results.append(cv_results)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Plot the model comparison
fig, ax = plt.subplots(figsize=(12, 10))
plt.title('Comparison of Classification Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('ROC-AUC Score')
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Parameters to tune for XGBoost
param_grids = [
    {'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100]},
    {'max_depth': [3, 4, 5], 'subsample': [0.7, 0.8]},
    {'colsample_bytree': [0.7, 0.8], 'colsample_bylevel': [0.7, 0.8]},
    {'scale_pos_weight': [0.5, 1], 'min_child_weight': [1, 2]},
    {'gamma': [0, 0.1, 0.2], 'reg_alpha': [0, 0.1]}
]

# Hyperparameter tuning for XGBoost with graphs for each set of parameters
for i, param_grid in enumerate(param_grids):
    grid_search = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, 
                               cv=3, n_jobs=-1, scoring='roc_auc', verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    print(f"Best parameters for set {i+1}: {grid_search.best_params_}")
    print(f"Best score for set {i+1}: {grid_search.best_score_}")

    # Plotting the results
    means = grid_search.cv_results_['mean_test_score']
    params = grid_search.cv_results_['params']
    plt.figure(figsize=(12, 6))
    plt.title(f"GridSearchCV results for parameter set {i+1}")
    plt.xlabel("Parameters")
    plt.ylabel("Mean Score")
    plt.bar(range(len(params)), means, color='b')
    plt.xticks(range(len(params)), params, rotation=90)
    plt.show()
