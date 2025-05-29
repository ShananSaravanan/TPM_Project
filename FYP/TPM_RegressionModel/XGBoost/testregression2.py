import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the saved model
model = joblib.load("xgb_rul_model.pkl")

# Load your test set (from training dataset or wherever you want)
test_df = pd.read_csv("AzureDataset/training_dataset_with_rul.csv")

# Create test split or just take a chunk for prediction
test_sample = test_df.sample(5, random_state=42)  # or however you like

# Match the feature columns exactly like in training
top_features = [
    'volt_mean_3h', 'rotate_mean_3h', 'pressure_mean_3h', 'vibration_mean_3h',
    'volt_mean_24h', 'rotate_mean_24h', 'pressure_mean_24h', 'vibration_mean_24h',
    'volt', 'rotate', 'pressure', 'vibration', 'age'
]

X_test = test_sample[top_features]
y_test = test_sample['RUL_hours']

# Apply the same scaler used during training to the test data
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)  # Make sure to use the same scaler or load the saved scaler if available

# Predict
predictions = model.predict(X_test_scaled)

# Show result
for i, pred in enumerate(predictions):
    print(f"Sample {i+1} - Predicted RUL: {pred:.2f} hours")

actual_rul = test_sample['RUL_hours'].values
for i, (pred, actual) in enumerate(zip(predictions, actual_rul)):
    print(f"Sample {i+1} - Predicted RUL: {pred:.2f} hours | Actual RUL: {actual} hours")

# Plotting Results
plt.figure(figsize=(8,5))
plt.plot(actual_rul, label='Actual RUL', marker='o')
plt.plot(predictions, label='Predicted RUL', marker='x')
plt.title("RUL Prediction vs Actual")
plt.xlabel("Sample Index")
plt.ylabel("RUL (hours)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
