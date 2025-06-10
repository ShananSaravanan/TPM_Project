import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Check XGBoost version
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
machines = pd.read_sql("SELECT * FROM machines", engine)
maint = pd.read_sql("SELECT * FROM maintenance", engine)
errors = pd.read_sql("SELECT * FROM errors", engine)

# Preprocessing
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
failures['datetime'] = pd.to_datetime(failures['datetime'])
maint['datetime'] = pd.to_datetime(maint['datetime'])
errors['datetime'] = pd.to_datetime(errors['datetime'])

# ... (previous code unchanged until failure merge)

# Merge failure timestamps (select earliest failure after telemetry datetime)
def get_next_failure(row, failures):
    future_failures = failures[(failures['machineid'] == row['machineid']) & (failures['datetime'] >= row['datetime'])]
    if not future_failures.empty:
        return future_failures['datetime'].min()
    return pd.NaT

data = telemetry.merge(machines, on='machineid', how='left')
data['last_failure'] = data.apply(lambda row: get_next_failure(row, failures), axis=1)

# Exclude machines with no failure data
data = data.dropna(subset=['last_failure'])

# Add maintenance and error features
data = data.merge(maint.groupby('machineid')['datetime'].max().reset_index(name='last_maint'), on='machineid', how='left')
data['time_since_last_maint'] = (data['datetime'] - data['last_maint']).dt.total_seconds() / 3600
data['no_maint'] = data['last_maint'].isna().astype(int)
data['time_since_last_maint'] = data['time_since_last_maint'].fillna(10000)

error_counts = errors.groupby('machineid').size().reset_index(name='error_count')
data = data.merge(error_counts, on='machineid', how='left')
data['error_count'] = data['error_count'].fillna(0)

data = data.merge(errors.groupby('machineid')['datetime'].max().reset_index(name='last_error'), on='machineid', how='left')
data['time_since_last_error'] = (data['datetime'] - data['last_error']).dt.total_seconds() / 3600
data['no_error'] = data['last_error'].isna().astype(int)
data['time_since_last_error'] = data['time_since_last_error'].fillna(10000)

# Calculate RUL (in hours)
data['RUL'] = (data['last_failure'] - data['datetime']).dt.total_seconds() / 3600
data = data[data['RUL'] <= 4000]  # Filter out RUL > 4000 hours
data = data[data['RUL'] >= 0]  # Remove invalid RUL

# Debug: Print RUL distribution
print("RUL Distribution (hours):")
print(data['RUL'].describe())

# Transform RUL: Square root
data['RUL'] = np.sqrt(data['RUL'])

# Normalize sensor features and new derived features
scaler = StandardScaler()
sensor_features = ['volt', 'rotate', 'pressure', 'vibration']
data[sensor_features] = scaler.fit_transform(data[sensor_features])

# Add derived features
data['sensor_degradation'] = data[sensor_features].mean(axis=1)
data['error_rate'] = data['error_count'] / (data['time_since_last_error'].replace(0, 1))
data['volt_change'] = data.groupby('machineid')['volt'].diff().fillna(0)
data['rotate_change'] = data.groupby('machineid')['rotate'].diff().fillna(0)
data['pressure_change'] = data.groupby('machineid')['pressure'].diff().fillna(0)
data['vibration_change'] = data.groupby('machineid')['vibration'].diff().fillna(0)
data['volt_error_inter'] = data['volt_change'] * data['error_rate']
data['rotate_error_inter'] = data['rotate_change'] * data['error_rate']
# New features
data['sensor_volatility'] = data[sensor_features].std(axis=1)
data['recent_error_count'] = data.groupby('machineid')['time_since_last_error'].transform(lambda x: (x < 24).sum())

# Normalize new features
new_features = ['sensor_degradation', 'error_rate', 'volt_change', 'rotate_change', 'pressure_change',
                'vibration_change', 'volt_error_inter', 'rotate_error_inter', 'sensor_volatility', 'recent_error_count']
data[new_features] = scaler.fit_transform(data[new_features])

# SPC: Flag outliers
for feature in sensor_features:
    stats = data.groupby('machineid')[feature].agg(['mean', 'std']).reset_index()
    stats.columns = ['machineid', f'{feature}_mean', f'{feature}_std']
    data = data.merge(stats, on='machineid', how='left')
    data[f'{feature}_outlier'] = ((data[feature] < data[f'{feature}_mean'] - 3 * data[f'{feature}_std']) |
                                 (data[feature] > data[f'{feature}_mean'] + 3 * data[f'{feature}_std'])).astype(int)

# Add rolling features with multiple windows
for feature in sensor_features:
    # Very short-term window (6 hours)
    data[f'{feature}_rolling_mean_6'] = data.groupby('machineid')[feature].rolling(window=6).mean().reset_index(level=0, drop=True)
    data[f'{feature}_rolling_std_6'] = data.groupby('machineid')[feature].rolling(window=6).std().reset_index(level=0, drop=True)
    # Short-term window (12 hours)
    data[f'{feature}_rolling_mean_12'] = data.groupby('machineid')[feature].rolling(window=12).mean().reset_index(level=0, drop=True)
    data[f'{feature}_rolling_std_12'] = data.groupby('machineid')[feature].rolling(window=12).std().reset_index(level=0, drop=True)
    # Medium-term window (24 hours)
    data[f'{feature}_rolling_mean_24'] = data.groupby('machineid')[feature].rolling(window=24).mean().reset_index(level=0, drop=True)
    data[f'{feature}_rolling_std_24'] = data.groupby('machineid')[feature].rolling(window=24).std().reset_index(level=0, drop=True)

# Normalize rolling features
rolling_features = [f'{feat}_rolling_{stat}_{win}' for feat in sensor_features for stat in ['mean', 'std'] for win in [6, 12, 24]]
data[rolling_features] = scaler.fit_transform(data[rolling_features].fillna(0))

# Encode model column
data = pd.get_dummies(data, columns=['model'], prefix='model')

# Drop temporary columns and handle NaN
data = data.drop(columns=[f'{f}_mean' for f in sensor_features] + [f'{f}_std' for f in sensor_features])
data = data.dropna()

# Prepare features and target
features = ['volt', 'rotate', 'pressure', 'vibration', 'age', 'time_since_last_maint', 'no_maint',
            'error_count', 'time_since_last_error', 'no_error', 'sensor_degradation', 'error_rate',
            'volt_change', 'rotate_change', 'pressure_change', 'vibration_change', 'volt_error_inter',
            'rotate_error_inter', 'sensor_volatility', 'recent_error_count'] + \
           rolling_features + \
           ['volt_outlier', 'rotate_outlier', 'pressure_outlier', 'vibration_outlier'] + \
           [col for col in data.columns if col.startswith('model_')]
target = 'RUL'

# Include datetime and machineID in X for saving, but exclude from training features
X = data[['datetime', 'machineid'] + features]
y = data[target]

# Stratified sampling based on RUL bins
data['RUL_bin'] = pd.qcut(data['RUL'], q=4, labels=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=data.loc[X.index, 'RUL_bin'], shuffle=True)
data = data.drop('RUL_bin', axis=1)

# Save test data with datetime and machineID
X_test.to_csv('X_test.csv', index=False)
y_test_transformed = pd.DataFrame({'RUL': y_test})  # Save transformed (sqrt) RUL
y_test_transformed.to_csv('y_test.csv', index=False)
print("✅ X_test and y_test saved to 'X_test.csv' and 'y_test.csv' with datetime and machineID")

# Select only model features for training and prediction
X_train_features = X_train[features]
X_test_features = X_test[features]

# Calculate sample weights (balanced with reduced emphasis on small RULs)
weights = np.log1p(np.maximum(y_train, 1))  # Logarithmic scaling to balance weights
weights = 1 / weights  # Inverse weighting
weights = weights / weights.mean()  # Normalize

# Debug: Check weights and predictions
print("Sample Weights Summary:")
print(pd.Series(weights).describe())

# Grid Search for hyperparameter tuning
param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [4, 5, 6],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8]
}
model = xgb.XGBRegressor(reg_lambda=1.0, reg_alpha=0.5, min_child_weight=1)
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
grid_search.fit(X_train_features, y_train)
print("Best parameters:", grid_search.best_params_)
final_model = grid_search.best_estimator_

# Evaluate model (on sqrt scale)
y_pred = final_model.predict(X_test_features)
y_pred = np.maximum(y_pred, 0)  # Enforce non-negative predictions

# Debug: Check predictions
print("Predicted RUL (sqrt scale) Summary:")
print(pd.Series(y_pred).describe())
print("Actual RUL (sqrt scale) Summary:")
print(pd.Series(y_test).describe())

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"✅ Model trained and saved. RMSE (sqrt scale): {rmse:.2f}, R² (sqrt scale): {r2:.2f}")

# Evaluate on original scale
y_pred_orig = y_pred ** 2  # Reverse square root transformation
y_test_orig = y_test ** 2

# Debug: Check original scale values
print("Predicted RUL (original scale) Summary:")
print(pd.Series(y_pred_orig).describe())
print("Actual RUL (original scale) Summary:")
print(pd.Series(y_test_orig).describe())

rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
r2_orig = r2_score(y_test_orig, y_pred_orig)
print(f"✅ Model evaluated on original scale. RMSE: {rmse_orig:.2f} hours, R²: {r2_orig:.2f}")

# Save model
joblib.dump(final_model, 'xgboost_rul_spc_model.pkl')

# Plot feature importance
xgb.plot_importance(final_model, max_num_features=10)
plt.title('Feature Importance')
plt.show()

# Plot predicted vs actual (sqrt scale)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Sqrt(RUL)')
plt.ylabel('Predicted Sqrt(RUL)')
plt.title('Predicted vs Actual RUL (Sqrt Scale)')
plt.show()

# Plot residuals
residuals = y_test_orig - y_pred_orig
plt.scatter(y_pred_orig, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted RUL (hours)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True)
plt.show()