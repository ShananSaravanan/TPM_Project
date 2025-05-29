import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import joblib
from tqdm import tqdm

# Load datasets
telemetry = pd.read_csv("AzureDataset/PdM_telemetry.csv", parse_dates=['datetime'])
failures = pd.read_csv("AzureDataset/PdM_failures.csv", parse_dates=['datetime'])

# Sort failures
failures = failures.sort_values(['machineID', 'datetime'])

# Step 1: Compute RUL
def compute_rul(telemetry, failures):
    telemetry = telemetry.copy()
    telemetry['RUL'] = float('inf')
    tqdm.pandas(desc="Computing RUL")

    for machine_id in telemetry['machineID'].unique():
        telemetry_m = telemetry[telemetry['machineID'] == machine_id].copy()
        failures_m = failures[failures['machineID'] == machine_id].sort_values('datetime')

        for fail_time in failures_m['datetime']:
            before_failure = telemetry_m[telemetry_m['datetime'] <= fail_time]
            rul_values = (fail_time - before_failure['datetime']).dt.total_seconds() / 3600
            telemetry.loc[before_failure.index, 'RUL'] = telemetry.loc[before_failure.index, 'RUL'].combine(rul_values, min)

    telemetry['RUL'] = telemetry['RUL'].clip(upper=200)
    return telemetry

# Step 2: Label RUL
def label_rul(df):
    bins = [-1, 24, 72, float('inf')]
    labels = ['critical', 'monitor', 'healthy']
    df['maintenance_status'] = pd.cut(df['RUL'], bins=bins, labels=labels)
    return df

# Main function
def main():
    # Compute and label
    telemetry_rul = compute_rul(telemetry, failures)
    telemetry_labeled = label_rul(telemetry_rul)
    telemetry_labeled = telemetry_labeled.dropna(subset=['maintenance_status'])

    # Features and labels
    X = telemetry_labeled[['volt', 'rotate', 'pressure', 'vibration']]
    y = telemetry_labeled['maintenance_status']
    label_map = {'critical': 0, 'monitor': 1, 'healthy': 2}
    y_encoded = y.map(label_map)

    # Balance classes
    rus = RandomUnderSampler()
    X_resampled, y_resampled = rus.fit_resample(X, y_encoded)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')

    # Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, 'xgb_rul_model.pkl')

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Predict full dataset
    full_X = telemetry_labeled[['volt', 'rotate', 'pressure', 'vibration']]
    full_scaled = scaler.transform(full_X)
    full_preds = model.predict(full_scaled)

    # Reverse label map
    inv_label_map = {0: 'critical', 1: 'monitor', 2: 'healthy'}
    telemetry_labeled['predicted_status'] = pd.Series(full_preds).map(inv_label_map)

    # Save to Excel
    output = telemetry_labeled[['datetime', 'machineID', 'volt', 'rotate', 'pressure', 'vibration', 'RUL', 'predicted_status']]
    output.to_excel("rul_predictions_output.xlsx", index=False)
    print("✅ Predictions saved to 'rul_predictions_output.xlsx'.")

if __name__ == "__main__":
    main()
