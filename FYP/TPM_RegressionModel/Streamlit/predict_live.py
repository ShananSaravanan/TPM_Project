# predict_live.py
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import time
from model_predictor import predict_rul
import streamlit as st

db_url = st.secrets["db_url"]

engine = create_engine(db_url)
def compute_derived_features(data):
    # Ensure datetime is in a consistent format
    data['datetime'] = pd.to_datetime(data['datetime'])

    # Merge maintenance data
    maint = pd.read_sql("SELECT * FROM maintenance", engine)
    maint_max = maint.groupby('machineid')['datetime'].max().reset_index(name='last_maint')
    data = data.merge(maint_max, on='machineid', how='left')
    data['last_maint'] = pd.to_datetime(data['last_maint'], errors='coerce')
    data['time_since_last_maint'] = (data['datetime'] - data['last_maint']).dt.total_seconds() / 3600
    data['no_maint'] = data['last_maint'].isna().astype(int)
    data['time_since_last_maint'] = data['time_since_last_maint'].fillna(10000)

    # Merge error data
    errors = pd.read_sql("SELECT * FROM errors", engine)
    error_counts = errors.groupby('machineid').size().reset_index(name='error_count')
    data = data.merge(error_counts, on='machineid', how='left')
    data['error_count'] = data['error_count'].fillna(0)
    error_max = errors.groupby('machineid')['datetime'].max().reset_index(name='last_error')
    data = data.merge(error_max, on='machineid', how='left')
    data['last_error'] = pd.to_datetime(data['last_error'], errors='coerce')
    data['time_since_last_error'] = (data['datetime'] - data['last_error']).dt.total_seconds() / 3600
    data['no_error'] = data['last_error'].isna().astype(int)
    data['time_since_last_error'] = data['time_since_last_error'].fillna(10000)

    # Use existing age from initial merge, or set default if missing
    if 'age' not in data.columns and 'age_x' in data.columns:
        data = data.rename(columns={'age_x': 'age'})  # Rename if merged as age_x
    elif 'age' not in data.columns:
        print("Warning: Age column not found, using default value of 3000")
        data['age'] = 3000  # Fallback default value
    data['age'] = data['age'].fillna(3000)  # Fallback if age is NaN

    # Compute changes (requires sorting by datetime and machineid)
    data = data.sort_values(['machineid', 'datetime'])
    for feature in ['volt', 'rotate', 'pressure', 'vibration']:
        data[feature + '_change'] = data.groupby('machineid')[feature].diff().fillna(0)

    # Compute rolling features for all required windows
    sensor_features = ['volt', 'rotate', 'pressure', 'vibration']
    windows = [6, 12, 24, 72]
    for feature in sensor_features:
        for window in windows:
            data[feature + '_rolling_mean_' + str(window)] = data.groupby('machineid')[feature].rolling(window=min(10, window), min_periods=1).mean().reset_index(level=0, drop=True)
            data[feature + '_rolling_std_' + str(window)] = data.groupby('machineid')[feature].rolling(window=min(10, window), min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

    # Compute sensor_volatility and recent_error_count
    data['sensor_volatility'] = data[sensor_features].std(axis=1)
    data['recent_error_count'] = data.groupby('machineid')['time_since_last_error'].transform(lambda x: (x < 24).sum())

    # Compute outliers (simplified SPC) with aligned indices and debugging
    for feature in sensor_features:
        # Group and compute stats with correct aggregation and renaming in one step
        group_stats = data.groupby('machineid')[feature].agg(
            **{f'{feature}_mean': 'mean', f'{feature}_std': 'std'}
        ).reset_index()
        print(f"Group stats columns for {feature}: {group_stats.columns.tolist()}")  # Debug print
        # Merge with original data
        data = data.merge(group_stats, on='machineid', how='left', suffixes=('', '_stats'))
        print(f"Data columns after merge for {feature}: {data.columns.tolist()}")  # Debug print
        # Access the merged columns
        mean_val = data[f'{feature}_mean']
        std_val = data[f'{feature}_std'].fillna(0)  # Handle cases with no variance
        data[feature + '_outlier'] = ((data[feature] < mean_val - 3 * std_val) | (data[feature] > mean_val + 3 * std_val)).astype(int)
        # Drop temporary columns
        data = data.drop(columns=[f'{feature}_mean', f'{feature}_std'])

    # Merge model data for one-hot encoding (only if not already present)
    if 'model' in data.columns:
        data = pd.get_dummies(data, columns=['model'], prefix='model')
    else:
        machines = pd.read_sql("SELECT machineid, model FROM machines", engine)
        data = data.merge(machines, on='machineid', how='left')
        data = pd.get_dummies(data, columns=['model'], prefix='model')

    return data.reset_index(drop=True)  # Reset index to align data

while True:
    try:
        # Fetch latest telemetry
        telemetry = pd.read_sql("SELECT * FROM telemetry ORDER BY datetime DESC LIMIT 10", engine)
        machines = pd.read_sql("SELECT * FROM machines", engine)
        data = telemetry.merge(machines, on='machineid', how='left')
        print(f"Initial columns after telemetry merge: {data.columns.tolist()}")  # Debug print

        # Compute derived features
        data = compute_derived_features(data)

        # Predict RUL
        predictions_df = predict_rul(data)
        predictions_df['prediction_time'] = datetime.now()

        # Save predictions
        predictions_df[['machineid', 'prediction_time', 'rul_pred']].to_sql(
            'predictions', engine, if_exists='append', index=False
        )
        print("✅ Live prediction inserted.")
    except Exception as e:
        print("❌ Prediction failed:", e)

    time.sleep(10)  # Adjust refresh rate