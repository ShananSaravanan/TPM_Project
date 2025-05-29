import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from tqdm import tqdm

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

# Cap RUL to 300 hours
cap = 300
data['RUL'] = (data['last_failure'] - data['datetime']).dt.total_seconds() / 3600
data['RUL'] = data['RUL'].fillna(cap)               # No failure? Assume high RUL
data = data[data['RUL'] >= 0]
data['RUL'] = data['RUL'].clip(upper=cap)           # Apply cap

# Feature engineering
data['volt_rotate'] = data['volt'] * data['rotate']
data['pressure_vibration'] = data['pressure'] * data['vibration']

features = ['volt', 'rotate', 'pressure', 'vibration', 'age', 'volt_rotate', 'pressure_vibration']
target = 'RUL'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Manual TQDM Training Loop
n_rounds = 100
pbar = tqdm(total=n_rounds, desc="Training Progress")

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1,  # We override this in loop
    learning_rate=0.1
)

booster = None
for i in range(n_rounds):
    model.n_estimators = i + 1
    model.fit(X_train, y_train, xgb_model=booster, eval_set=[(X_test, y_test)], verbose=False)
    booster = model.get_booster()
    pbar.update(1)

pbar.close()

# Save model
joblib.dump(model, 'xgboost_rul_model.pkl')
print("✅ Model trained and saved.")
