import joblib
import pandas as pd
import numpy as np

# Load the new model
model = joblib.load("FYP/TPM_RegressionModel/XGBoost/V2_Postgre/Model/xgboost_rul_spc_model.pkl")

# Get the feature names from the trained model
feature_names = model.get_booster().feature_names

def compute_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetime is in a consistent format
    data = data.copy()
    data['datetime'] = pd.to_datetime(data['datetime'])

    # Handle maintenance-related features
    if 'time_since_last_maint' not in data.columns:
        data['time_since_last_maint'] = 10000  # Default for no maintenance
    if 'no_maint' not in data.columns:
        data['no_maint'] = 1  # Default for no maintenance

    # Handle error-related features
    if 'error_count' not in data.columns:
        data['error_count'] = 0
    if 'time_since_last_error' not in data.columns:
        data['time_since_last_error'] = 10000  # Default for no errors
    if 'no_error' not in data.columns:
        data['no_error'] = 1  # Default for no errors

    # Handle age
    if 'age' not in data.columns:
        print("Warning: Age column not found, using default value of 3000")
        data['age'] = 3000  # Fallback default value
    data['age'] = data['age'].fillna(3000)

    # Compute changes
    data = data.sort_values(['machineid', 'datetime'])
    for feature in ['volt', 'rotate', 'pressure', 'vibration']:
        data[feature + '_change'] = data.groupby('machineid')[feature].diff().fillna(0)

    # Compute rolling features
    sensor_features = ['volt', 'rotate', 'pressure', 'vibration']
    windows = [6, 12, 24, 72]
    for feature in sensor_features:
        for window in windows:
            data[feature + '_rolling_mean_' + str(window)] = data.groupby('machineid')[feature].rolling(window=min(10, window), min_periods=1).mean().reset_index(level=0, drop=True)
            data[feature + '_rolling_std_' + str(window)] = data.groupby('machineid')[feature].rolling(window=min(10, window), min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

    # Compute sensor_volatility and recent_error_count
    data['sensor_volatility'] = data[sensor_features].std(axis=1)
    data['recent_error_count'] = data.groupby('machineid')['time_since_last_error'].transform(lambda x: (x < 24).sum())

    # Compute outliers (simplified SPC)
    for feature in sensor_features:
        group_stats = data.groupby('machineid')[feature].agg(
            **{f'{feature}_mean': 'mean', f'{feature}_std': 'std'}
        ).reset_index()
        data = data.merge(group_stats, on='machineid', how='left', suffixes=('', '_stats'))
        mean_val = data[f'{feature}_mean']
        std_val = data[f'{feature}_std'].fillna(0)
        data[feature + '_outlier'] = ((data[feature] < mean_val - 3 * std_val) | (data[feature] > mean_val + 3 * std_val)).astype(int)
        data = data.drop(columns=[f'{feature}_mean', f'{feature}_std'])

    # Handle model one-hot encoding
    if 'model' in data.columns:
        data = pd.get_dummies(data, columns=['model'], prefix='model')
    else:
        # Default to model1 if no model info provided
        data['model_model1'] = 1
        data['model_model2'] = 0
        data['model_model3'] = 0
        data['model_model4'] = 0

    return data.reset_index(drop=True)

def predict_rul(data: pd.DataFrame) -> pd.DataFrame:
    # Create a copy to avoid modifying the original data
    data_copy = data.copy()

    # Compute derived features
    data_copy = compute_derived_features(data_copy)

    # Ensure all required features are present
    required_features = [
        'volt', 'rotate', 'pressure', 'vibration', 'age', 'time_since_last_maint', 'no_maint',
        'error_count', 'time_since_last_error', 'no_error', 'sensor_degradation', 'error_rate',
        'volt_change', 'rotate_change', 'pressure_change', 'vibration_change', 'volt_error_inter',
        'rotate_error_inter', 'volt_rolling_mean_24', 'rotate_rolling_mean_24', 'pressure_rolling_mean_24',
        'vibration_rolling_mean_24', 'volt_rolling_std_24', 'rotate_rolling_std_24', 'pressure_rolling_std_24',
        'vibration_rolling_std_24', 'volt_rolling_mean_72', 'rotate_rolling_mean_72', 'pressure_rolling_mean_72',
        'vibration_rolling_mean_72', 'volt_rolling_std_72', 'rotate_rolling_std_72', 'pressure_rolling_std_72',
        'vibration_rolling_std_72', 'volt_outlier', 'rotate_outlier', 'pressure_outlier', 'vibration_outlier',
        'model_model1', 'model_model2', 'model_model3', 'model_model4'
    ]

    # Compute additional derived features if not already present
    if 'sensor_degradation' not in data_copy.columns:
        data_copy['sensor_degradation'] = data_copy[['volt', 'rotate', 'pressure', 'vibration']].mean(axis=1)
    if 'error_rate' not in data_copy.columns:
        data_copy['error_rate'] = data_copy['error_count'] / data_copy['time_since_last_error'].replace(0, 1)
    if 'volt_error_inter' not in data_copy.columns:
        data_copy['volt_error_inter'] = data_copy['volt_change'] * data_copy['error_rate']
    if 'rotate_error_inter' not in data_copy.columns:
        data_copy['rotate_error_inter'] = data_copy['rotate_change'] * data_copy['error_rate']

    # Add missing features with zeros
    for feature in required_features:
        if feature not in data_copy.columns:
            data_copy[feature] = 0

    # Align features with the model's expected order
    X = data_copy[feature_names]

    # Predict RUL (sqrt scale)
    rul_pred_sqrt = model.predict(X)

    # Create a result DataFrame with original data and predictions
    result = data.copy()  # Preserve original data
    result['rul_pred'] = rul_pred_sqrt ** 2  # Reverse square root transformation

    # Return relevant columns
    return result[['machineid', 'datetime', 'rul_pred']]

if __name__ == "__main__":
    # Example usage (for testing)
    sample_data = pd.DataFrame({
        'machineid': [99], 'datetime': [pd.Timestamp.now()], 'volt': [162], 'rotate': [453.50],
        'pressure': [95], 'vibration': [34], 'age': [14], 'time_since_last_maint': [120],
        'no_maint': [0], 'error_count': [16], 'time_since_last_error': [144], 'no_error': [0],
        'model': ['model1']
    })
    result = predict_rul(sample_data)
    print(result)