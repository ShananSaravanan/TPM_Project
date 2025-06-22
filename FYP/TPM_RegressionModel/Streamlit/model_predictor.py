# model_predictor.py
import joblib
import pandas as pd
import numpy as np

# Load the new model
model = joblib.load("FYP/TPM_RegressionModel/XGBoost/V2_Postgre/Model/xgboost_rul_spc_model.pkl")

# Get the feature names from the trained model
feature_names = model.get_booster().feature_names

def predict_rul(data: pd.DataFrame) -> pd.DataFrame:
    # Create a copy to avoid modifying the original data
    data_copy = data.copy()

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

    # Compute derived features if not already present
    if 'sensor_degradation' not in data_copy.columns:
        data_copy['sensor_degradation'] = data_copy[['volt', 'rotate', 'pressure', 'vibration']].mean(axis=1)
    if 'error_rate' not in data_copy.columns and 'error_count' in data_copy.columns and 'time_since_last_error' in data_copy.columns:
        data_copy['error_rate'] = data_copy['error_count'] / data_copy['time_since_last_error'].replace(0, 1)
    if 'volt_error_inter' not in data_copy.columns and 'volt_change' in data_copy.columns and 'error_rate' in data_copy.columns:
        data_copy['volt_error_inter'] = data_copy['volt_change'] * data_copy['error_rate']
    if 'rotate_error_inter' not in data_copy.columns and 'rotate_change' in data_copy.columns and 'error_rate' in data_copy.columns:
        data_copy['rotate_error_inter'] = data_copy['rotate_change'] * data_copy['error_rate']

    # Add missing rolling features with zeros (to be computed by predict_live.py)
    for feature in required_features:
        if feature not in data_copy.columns:
            data_copy[feature] = 0

    # Align features with the model's expected order
    X = data_copy[feature_names]

    # Predict RUL (sqrt scale)
    rul_pred_sqrt = model.predict(X)

    # Create a result DataFrame with original data and predictions
    result = data.copy()  # Preserve original data
    result['rul_pred'] = rul_pred_sqrt ** 2  # Reverse square root transformation to original scale

    # Return relevant columns
    return result[['machineid', 'datetime', 'rul_pred']]

if __name__ == "__main__":
    # Example usage (for testing)
    sample_data = pd.DataFrame({
        'machineid': [1], 'datetime': [pd.Timestamp.now()], 'volt': [158], 'rotate': [429.50],
        'pressure': [94], 'vibration': [58], 'age': [18], 'time_since_last_maint': [0],
        'no_maint': [1], 'error_count': [3], 'time_since_last_error': [24], 'no_error': [0]
    })
    result = predict_rul(sample_data)
    print(result)