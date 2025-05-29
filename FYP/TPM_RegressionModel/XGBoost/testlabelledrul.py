import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load("xgb_rul_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load your new test data
df = pd.read_csv("AzureDataset/test_set.csv", parse_dates=['datetime'])

# Select feature columns
features = ['volt', 'rotate', 'pressure', 'vibration']
X_new = df[features]

# Apply the same scaling as training
X_new_scaled = scaler.transform(X_new)

# Predict
predictions = model.predict(X_new_scaled)

# Map predictions back to labels
label_map_reverse = {0: 'critical', 1: 'monitor', 2: 'healthy'}
df['predicted_status'] = pd.Series(predictions).map(label_map_reverse)

# Save output
df.to_csv("test_sample_predictions.csv", index=False)
print("Predictions saved to test_sample_predictions.csv")
