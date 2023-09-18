import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#load CSV
df = pd.read_csv('click_fraud_data.csv')


#Categorical Feature Summary

# Selecting columns
df1 = df.iloc[:, 1:6]

# Counting unique values in each column (this assumes df1 is a DataFrame)
df_ead = pd.DataFrame(df1.nunique()).reset_index()
df_ead.columns = ['feature', 'level_count']

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='feature', y='level_count', data=df_ead, palette='tab10')

# Adding text labels
for i, count in enumerate(df_ead['level_count']):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.title('Categorical Feature Summary')
plt.xlabel('Categorical Feature')
plt.ylabel('Number of Cardinality')
plt.show()


------

# Distribution by Features

def plot_top_n(df, column_name, n=10, use_log_scale=False):
    df_grouped = df.groupby(column_name).size().reset_index(name='count')
    df_sorted = df_grouped.nlargest(n, 'count')
    
    plt.figure(figsize=(10, 6))
    plt.barh(df_sorted[column_name], df_sorted['count'], color='skyblue')
    
    if use_log_scale:
        plt.xscale('log')
        plt.xlabel('Count (Log Scale)')
    else:
        plt.xlabel('Count')
        
    plt.ylabel(column_name)
    plt.title(f'Distribution of {column_name}')
    plt.show()

# Plot top 10 of each category
plot_top_n(df, 'device')
plot_top_n(df, 'app')
plot_top_n(df, 'ip')
plot_top_n(df, 'channel')
plot_top_n(df, 'os')

-------------

# App downloaded percentage 

# Calculating the proportion of each class in the 'is_attributed' column
mean = (df.is_attributed.values == 1).mean()
not_mean = 1 - mean

# Plotting the bar chart with percentages
plt.figure(figsize=(6, 6))
ax = sns.barplot(['App Downloaded (1)', 'Not Downloaded (0)'], [mean, not_mean])
ax.set(ylabel='Proportion', title='App Downloaded vs Not Downloaded')

# Adding text labels on the bars
for p, uniq in zip(ax.patches, [mean, not_mean]):
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.01,
            '{}%'.format(round(uniq * 100, 2)),
            ha="center")
plt.show()

--------------

# Time Series Click/Fraud Distribution

# Convert click_time to datetime format
df['click_time'] = pd.to_datetime(df['click_time'])

# Group by click_time and summarize
df_series = df.groupby('click_time').agg({'is_attributed': ['count', 'sum']}).reset_index()
df_series.columns = ['click_time', 'click_count', 'fraud_count']

# Time Series Distribution of Click Records
plt.figure(figsize=(12, 6))
plt.plot(df_series['click_time'], df_series['click_count'], color='darkorange')
plt.title('Time Series Click Distribution', fontsize=16)
plt.xlabel('Click Time', fontsize=12)
plt.ylabel('Click Count per Second', fontsize=12)
plt.grid(True)
plt.show()

# Time Series Distribution of Fraud Clicks
plt.figure(figsize=(12, 6))
plt.plot(df_series['click_time'], df_series['fraud_count'], color='darkblue')
plt.title('Time Series Fraud Distribution', fontsize=16)
plt.xlabel('Click Time', fontsize=12)
plt.ylabel('Fraud Count per Second', fontsize=12)
plt.grid(True)
plt.show()

----------------

# Hourly Click Ratio/Converstion

# Convert 'click_time' and 'attributed_time' columns to datetime format
df['click_time_dt'] = pd.to_datetime(df['click_time'])
df['attributed_time_dt'] = pd.to_datetime(df['attributed_time'])

# Round 'click_time' to the nearest hour and store it in a new column
df['click_time_rounded'] = df['click_time_dt'].dt.round('H')

# Plot the frequency of clicks per hour
hourly_clicks = df[['click_time_rounded', 'is_attributed']].groupby(['click_time_rounded'], as_index=True).count()
hourly_clicks.plot()
plt.title('Hourly Click Frequency')
plt.ylabel('Number of Clicks')

# Plot the conversion ratio per hour
hourly_conversion_ratio = df[['click_time_rounded', 'is_attributed']].groupby(['click_time_rounded'], as_index=True).mean()
hourly_conversion_ratio.plot()
plt.title('Hourly Conversion Ratio')
plt.ylabel('Conversion Ratio')


-----------------

# Heat Map


# Heatmap to Show Correlations Among Numeric Features
selected_columns = ['channel', 'os', 'device', 'app', 'ip', 'is_attributed']

# Remove the selected columns to focus on numerical ones for correlation
data_for_correlation = df_total.drop(selected_columns, axis=1)

# Calculate the correlation matrix
correlation_matrix = data_for_correlation.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
heatmap_plot = sns.heatmap(data=correlation_matrix)
plt.title('Correlation Heatmap of Numeric Features')


-----------------------
# Class Balancing 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Separate features and target variable from the training set
features = train.drop(columns=['is_attributed'])
target = train['is_attributed']

# Balance the dataset using Random Under-Sampling
undersampler = RandomUnderSampler(random_state=1234)
features_balanced, target_balanced = undersampler.fit_resample(features, target)

# Remove the 'click_time' column as it's not needed
features_balanced = features_balanced.drop(columns=['click_time'])

# Standardize the features
std_scaler = StandardScaler()
features_standardized = std_scaler.fit_transform(features_balanced)

# Handle any infinity or NaN values
features_standardized[np.isinf(features_standardized)] = 0
features_standardized = np.nan_to_num(features_standardized)

# Perform PCA to reduce dimensionality while retaining 95% variance
pca_model = PCA(0.95)
features_reduced = pca_model.fit_transform(features_standardized)

# Print the number of principal components used
print(f"Number of principal components: {pca_model.n_components_}")



# Balance Chart 
# Importing necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Generating synthetic data to represent the original class distribution
# This is just a placeholder; you would replace this with your real data
y_original = [0]*9965 + [1]*35

# Generating synthetic data to represent the balanced class distribution
# This is also a placeholder; you would replace this with your balanced data
y_balanced = [0]*5000 + [1]*5000

# Function to plot the class distribution
def plot_class_distribution(y, title):
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    total_count = len(y)
    for p in plt.gca().patches:
        height = p.get_height()
        plt.gca().text(p.get_x() + p.get_width()/2, height, '{} ({}%)'.format(height, round((height/total_count)*100, 2)), ha='center')

# Plotting the original class distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_class_distribution(y_original, 'Original Class Distribution')

# Plotting the balanced class distribution
plt.subplot(1, 2, 2)
plot_class_distribution(y_balanced, 'Balanced Class Distribution')

plt.tight_layout()
plt.show()

