import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Print XGBoost version
print("XGBoost version:", xgb.__version__)

# Connect to PostgreSQL
db_config = {
    'host': 'localhost',
    'port': '5433',
    'database': 'AzureTPMDB',
    'user': 'postgres',
    'password': 'root'
}
conn_str = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
engine = create_engine(conn_str)

# Load DataFrames
telemetry = pd.read_sql("SELECT * FROM telemetry", engine)
failures = pd.read_sql("SELECT * FROM failures", engine)

# Preprocessing: Convert datetime columns
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
failures['datetime'] = pd.to_datetime(failures['datetime'])

# Step 1: Identify the most significant feature (for reference)
telemetry['failure_label'] = 0
for _, failure in failures.iterrows():
    mask = (telemetry['machineid'] == failure['machineid']) & \
           (telemetry['datetime'].between(failure['datetime'] - pd.Timedelta(hours=1), failure['datetime']))
    telemetry.loc[mask, 'failure_label'] = 1

# Train XGBoost classifier to find feature importance
features = ['volt', 'rotate', 'pressure', 'vibration']
X = telemetry[features]
y = telemetry['failure_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf_model = xgb.XGBClassifier(random_state=42)
clf_model.fit(X_train, y_train)

# Extract and plot feature importance
feature_importance = pd.Series(clf_model.feature_importances_, index=features).sort_values(ascending=False)
print("Feature Importance Scores:")
print(feature_importance)
most_significant_feature = 'rotate'  # Fixed to rotate
print(f"Selected Feature: {most_significant_feature}")

plt.figure(figsize=(8, 6))
feature_importance.plot(kind='bar')
plt.title('Feature Importance for Failure Prediction')
plt.ylabel('Importance Score (Gain)')
plt.tight_layout()
plt.show()

# Step 2: Prepare data using Approach 1 (telemetry between failures)
data = []
for machineid in failures['machineid'].unique():
    machine_telemetry = telemetry[telemetry['machineid'] == machineid].sort_values('datetime')
    machine_failures = failures[failures['machineid'] == machineid].sort_values('datetime')
    for i in range(len(machine_failures)):
        start_time = machine_failures.iloc[i-1]['datetime'] if i > 0 else machine_telemetry['datetime'].min()
        end_time = machine_failures.iloc[i]['datetime']
        window = machine_telemetry[(machine_telemetry['datetime'] >= start_time) & 
                                  (telemetry['datetime'] <= end_time)].copy()
        window['RUL'] = (end_time - window['datetime']).dt.total_seconds() / 3600
        data.append(window)

data = pd.concat(data, ignore_index=True)
data = data[data['RUL'] >= 0]

# Filter out extreme RUL values
data = data[data['RUL'] <= 4000]

# Debug: Print RUL distribution
print("\nRUL Distribution (hours):")
print(data['RUL'].describe())

# Transform RUL: Square root
data['RUL'] = np.sqrt(data['RUL'])

# Normalize the selected feature
scaler = StandardScaler()
data[most_significant_feature] = scaler.fit_transform(data[[most_significant_feature]])

# Add derived features
# Rolling statistics (6, 12, 24 hours)
for window in [6, 12, 24]:
    data[f'{most_significant_feature}_rolling_mean_{window}'] = data.groupby('machineid')[most_significant_feature]\
        .rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
    data[f'{most_significant_feature}_rolling_std_{window}'] = data.groupby('machineid')[most_significant_feature]\
        .rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True)

# Exponential moving average (span=6 hours)
data[f'{most_significant_feature}_ema_6'] = data.groupby('machineid')[most_significant_feature]\
    .ewm(span=6, adjust=False).mean().reset_index(level=0, drop=True)

# Change rate
data[f'{most_significant_feature}_change'] = data.groupby('machineid')[most_significant_feature].diff().fillna(0)

# SPC: Outlier flag (±2 std)
stats = data.groupby('machineid')[most_significant_feature].agg(['mean', 'std']).reset_index()
stats.columns = ['machineid', f'{most_significant_feature}_mean', f'{most_significant_feature}_std']
data = data.merge(stats, on='machineid', how='left')
data[f'{most_significant_feature}_outlier'] = (
    (data[most_significant_feature] < data[f'{most_significant_feature}_mean'] - 2 * data[f'{most_significant_feature}_std']) |
    (data[most_significant_feature] > data[f'{most_significant_feature}_mean'] + 2 * data[f'{most_significant_feature}_std'])
).astype(int)

# Normalize derived features
derived_features = [f'{most_significant_feature}_rolling_mean_{w}' for w in [6, 12, 24]] + \
                   [f'{most_significant_feature}_rolling_std_{w}' for w in [6, 12, 24]] + \
                   [f'{most_significant_feature}_ema_6', f'{most_significant_feature}_change', 
                    f'{most_significant_feature}_outlier']
data[derived_features] = scaler.fit_transform(data[derived_features].fillna(0))

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved to 'scaler.pkl'")

# Drop temporary columns
data = data.drop(columns=[f'{most_significant_feature}_mean', f'{most_significant_feature}_std'])

# Handle NaN
data = data.dropna()

# Prepare features and target
features = [most_significant_feature] + derived_features
target = 'RUL'

# Include datetime and machineid for saving
X = data[['datetime', 'machineid'] + features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test data
X_test.to_csv('X_test.csv', index=False)
y_test_transformed = pd.DataFrame({'RUL': y_test})
y_test_transformed.to_csv('y_test.csv', index=False)
print("✅ X_test and y_test saved to 'X_test.csv' and 'y_test.csv'")

# Select only model features for training
X_train_features = X_train[features]
X_test_features = X_test[features]

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'min_child_weight': [1, 3, 5]
}
model = xgb.XGBRegressor(reg_lambda=1.0, reg_alpha=0.5, objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train_features, y_train)
print("Best Parameters:", grid_search.best_params_)

# Train final model
final_model = grid_search.best_estimator_

# Save model
joblib.dump(final_model, 'xgboost_rul_model.pkl')
print("✅ Model saved to 'xgboost_rul_model.pkl'")

# Plot feature importance
xgb.plot_importance(final_model, max_num_features=10)
plt.title('Feature Importance for RUL Prediction')
plt.tight_layout()
plt.show()