import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

# Preprocessing
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
failures['datetime'] = pd.to_datetime(failures['datetime'])

failure_timestamps = failures.groupby('machineid')['datetime'].max().reset_index()
failure_timestamps.columns = ['machineid', 'last_failure']

data = telemetry.merge(failure_timestamps, on='machineid', how='left')
data = data.merge(machines, on='machineid', how='left')

# RUL and feature engineering
cap = 300
data['RUL'] = (data['last_failure'] - data['datetime']).dt.total_seconds() / 3600
data['RUL'] = data['RUL'].fillna(cap)
data = data[data['RUL'] >= 0]
data['RUL'] = data['RUL'].clip(upper=cap)

data['volt_rotate'] = data['volt'] * data['rotate']
data['pressure_vibration'] = data['pressure'] * data['vibration']

features = ['volt', 'rotate', 'pressure', 'vibration', 'age', 'volt_rotate', 'pressure_vibration']
target = 'RUL'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Load model
model = joblib.load('XGBoost/Models/xgboost_rul_model(postgre).pkl')

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 Evaluation Metrics:")
print(f"  - MAE : {mae:.2f} hours")
print(f"  - RMSE: {rmse:.2f} hours")
print(f"  - R²  : {r2:.4f}")

# Optional: Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([0, cap], [0, cap], color='red', linestyle='--')
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Predicted vs Actual RUL")
plt.grid(True)
plt.tight_layout()
plt.show()
