import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from tqdm import tqdm
from sklearn.utils import resample

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

# Add maintenance and error features
data = data.merge(maint.groupby('machineid')['datetime'].max().reset_index(name='last_maint'), on='machineid', how='left')
data['time_since_last_maint'] = (data['datetime'] - data['last_maint']).dt.total_seconds() / 3600
data['time_since_last_maint'] = data['time_since_last_maint'].fillna(0)

error_counts = errors.groupby('machineid').size().reset_index(name='error_count')
data = data.merge(error_counts, on='machineid', how='left')
data['error_count'] = data['error_count'].fillna(0)

data = data.merge(errors.groupby('machineid')['datetime'].max().reset_index(name='last_error'), on='machineid', how='left')
data['time_since_last_error'] = (data['datetime'] - data['last_error']).dt.total_seconds() / 3600
data['time_since_last_error'] = data['time_since_last_error'].fillna(0)

# Calculate RUL (in hours)
data['RUL'] = (data['last_failure'] - data['datetime']).dt.total_seconds() / 3600
data = data[data['RUL'].notnull()]  # Remove records with no failure
data = data[data['RUL'] < 1000]     # Cap RUL at 1000 hours
data['RUL'] = np.log1p(data['RUL'])  # Log transform RUL

# SPC: Apply Control Charts to Telemetry Features
features = ['volt', 'rotate', 'pressure', 'vibration']
for feature in features:
    stats = data.groupby('machineid')[feature].agg(['mean', 'std']).reset_index()
    stats.columns = ['machineid', f'{feature}_mean', f'{feature}_std']
    data = data.merge(stats, on='machineid', how='left')
    data[f'{feature}_lower'] = data[f'{feature}_mean'] - 3 * data[f'{feature}_std']
    data[f'{feature}_upper'] = data[f'{feature}_mean'] + 3 * data[f'{feature}_std']
    data[feature] = data[feature].clip(lower=data[f'{feature}_lower'], upper=data[f'{feature}_upper'])

# Add rolling features
for feature in ['volt', 'rotate', 'pressure', 'vibration']:
    data[f'{feature}_rolling_mean'] = data.groupby('machineid')[feature].rolling(window=24).mean().reset_index(level=0, drop=True)
    data[f'{feature}_rolling_std'] = data.groupby('machineid')[feature].rolling(window=24).std().reset_index(level=0, drop=True)

# SPC: Apply Control Charts to RUL (less aggressive)
rul_stats = data.groupby('machineid')['RUL'].agg(['mean', 'std']).reset_index()
rul_stats.columns = ['machineid', 'RUL_mean', 'RUL_std']
data = data.merge(rul_stats, on='machineid', how='left')
data['RUL_lower'] = data['RUL_mean'] - 3 * data['RUL_std']
data['RUL_upper'] = data['RUL_mean'] + 3 * data['RUL_std']
data['RUL'] = data['RUL'].clip(lower=data['RUL_lower'], upper=data['RUL_upper'])

# Drop temporary columns and handle NaN from rolling
data = data.drop(columns=[f'{f}_mean' for f in features] + [f'{f}_std' for f in features] +
                 [f'{f}_lower' for f in features] + [f'{f}_upper' for f in features] +
                 ['RUL_mean', 'RUL_std', 'RUL_lower', 'RUL_upper'])
data = data.dropna()

# Feature engineering
data['volt_rotate'] = data['volt'] * data['rotate']
data['pressure_vibration'] = data['pressure'] * data['vibration']

# Prepare features and target
features = ['volt', 'rotate', 'pressure', 'vibration', 'age', 'volt_rotate', 'pressure_vibration',
            'time_since_last_maint', 'error_count', 'time_since_last_error',
            'volt_rolling_mean', 'rotate_rolling_mean', 'pressure_rolling_mean', 'vibration_rolling_mean',
            'volt_rolling_std', 'rotate_rolling_std', 'pressure_rolling_std', 'vibration_rolling_std']
target = 'RUL'

X = data[features]
y = data[target]

# Oversample low RUL
low_rul = data[data['RUL'] < np.log1p(100)]
low_rul_upsampled = resample(low_rul, replace=True, n_samples=len(data) // 2, random_state=42)
data = pd.concat([data, low_rul_upsampled])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Save test data for evaluation
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("✅ X_test and y_test saved to 'X_test.csv' and 'y_test.csv'")

# Manual TQDM Training Loop
n_rounds = 100
pbar = tqdm(total=n_rounds, desc="Training Progress")

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1,  # Override in loop
    learning_rate=0.1
)

booster = None
for i in range(n_rounds):
    model.n_estimators = i + 1
    model.fit(X_train, y_train, xgb_model=booster, eval_set=[(X_test, y_test)], verbose=False)
    booster = model.get_booster()
    pbar.update(1)

pbar.close()
from sklearn.metrics import mean_squared_error, r2_score
# Evaluate model
y_pred = model.predict(X_test)
y_pred_orig = np.expm1(y_pred)  # Reverse log transformation
y_test_orig = np.expm1(y_test)  # Reverse log transformation
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
r2 = r2_score(y_test_orig, y_pred_orig)
print(f"✅ Model trained and saved. RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Save model
joblib.dump(model, 'xgboost_rul_spc_model.pkl')