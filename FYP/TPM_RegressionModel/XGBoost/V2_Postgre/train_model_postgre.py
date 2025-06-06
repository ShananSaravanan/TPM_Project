import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from tqdm import tqdm
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

# Merge failure timestamps
failure_timestamps = failures.groupby('machineid')['datetime'].max().reset_index()
failure_timestamps.columns = ['machineid', 'last_failure']
data = telemetry.merge(failure_timestamps, on='machineid', how='left')
data = data.merge(machines, on='machineid', how='left')

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

# Log transform RUL to reduce skewness
data['RUL'] = np.log1p(data['RUL'])

# Normalize sensor features
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
data['volt_error_inter'] = data['volt_change'] * data['error_rate']  # Interaction term
data['rotate_error_inter'] = data['rotate_change'] * data['error_rate']

# SPC: Flag outliers
for feature in sensor_features:
    stats = data.groupby('machineid')[feature].agg(['mean', 'std']).reset_index()
    stats.columns = ['machineid', f'{feature}_mean', f'{feature}_std']
    data = data.merge(stats, on='machineid', how='left')
    data[f'{feature}_outlier'] = ((data[feature] < data[f'{feature}_mean'] - 3 * data[f'{feature}_std']) |
                                 (data[feature] > data[f'{feature}_mean'] + 3 * data[f'{feature}_std'])).astype(int)

# Add rolling features with multiple windows
for feature in sensor_features:
    # Short-term window
    data[f'{feature}_rolling_mean_24'] = data.groupby('machineid')[feature].rolling(window=24).mean().reset_index(level=0, drop=True)
    data[f'{feature}_rolling_std_24'] = data.groupby('machineid')[feature].rolling(window=24).std().reset_index(level=0, drop=True)
    # Long-term window
    data[f'{feature}_rolling_mean_72'] = data.groupby('machineid')[feature].rolling(window=72).mean().reset_index(level=0, drop=True)
    data[f'{feature}_rolling_std_72'] = data.groupby('machineid')[feature].rolling(window=72).std().reset_index(level=0, drop=True)

# Encode model column
data = pd.get_dummies(data, columns=['model'], prefix='model')

# Drop temporary columns and handle NaN
data = data.drop(columns=[f'{f}_mean' for f in sensor_features] + [f'{f}_std' for f in sensor_features])
data = data.dropna()

# Prepare features and target
features = ['volt', 'rotate', 'pressure', 'vibration', 'age', 'time_since_last_maint', 'no_maint',
            'error_count', 'time_since_last_error', 'no_error', 'sensor_degradation', 'error_rate',
            'volt_change', 'rotate_change', 'pressure_change', 'vibration_change', 'volt_error_inter',
            'rotate_error_inter', 'volt_rolling_mean_24', 'rotate_rolling_mean_24', 'pressure_rolling_mean_24',
            'vibration_rolling_mean_24', 'volt_rolling_std_24', 'rotate_rolling_std_24', 'pressure_rolling_std_24',
            'vibration_rolling_std_24', 'volt_rolling_mean_72', 'rotate_rolling_mean_72', 'pressure_rolling_mean_72',
            'vibration_rolling_mean_72', 'volt_rolling_std_72', 'rotate_rolling_std_72', 'pressure_rolling_std_72',
            'vibration_rolling_std_72', 'volt_outlier', 'rotate_outlier', 'pressure_outlier', 'vibration_outlier'] + \
           [col for col in data.columns if col.startswith('model_')]
target = 'RUL'

X = data[features]
y = data[target]

# Stratified sampling based on RUL bins
data['RUL_bin'] = pd.qcut(data['RUL'], q=4, labels=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=data.loc[X.index, 'RUL_bin'], shuffle=True)
data = data.drop('RUL_bin', axis=1)

# Save test data
X_test.to_csv('X_test.csv', index=False)
y_test_transformed = pd.DataFrame({'RUL': y_test})
y_test_transformed.to_csv('y_test.csv', index=False)
print("✅ X_test and y_test saved to 'X_test.csv' and 'y_test.csv'")

# Train final model with adjusted parameters using xgb.train
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.15,
    'max_depth': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'reg_lambda': 0.5,
    'reg_alpha': 0.1
}
evals = [(dtrain, 'train'), (dtest, 'test')]
booster = xgb.train(
    params,
    dtrain,
    num_boost_round=150,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=False
)

# Wrap trained model in XGBRegressor for compatibility
final_model = xgb.XGBRegressor()
final_model._Booster = booster

# Evaluate model (on log scale)
y_pred = final_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"✅ Model trained and saved. RMSE (log scale): {rmse:.2f}, R² (log scale): {r2:.2f}")

# Evaluate on original scale
y_pred_orig = np.expm1(y_pred)
y_test_orig = np.expm1(y_test)
rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
r2_orig = r2_score(y_test_orig, y_pred_orig)
print(f"✅ Model evaluated on original scale. RMSE: {rmse_orig:.2f} hours, R²: {r2_orig:.2f}")

# Save model
joblib.dump(final_model, 'xgboost_rul_spc_model.pkl')

# Plot feature importance
xgb.plot_importance(final_model, max_num_features=10)
plt.title('Feature Importance')
plt.show()

# Plot predicted vs actual (log scale)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Log(RUL + 1)')
plt.ylabel('Predicted Log(RUL + 1)')
plt.title('Predicted vs Actual RUL (Log Scale)')
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