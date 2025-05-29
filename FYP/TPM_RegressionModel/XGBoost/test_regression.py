import pandas as pd
import xgboost as xgb
import joblib

# Load the saved model
model = joblib.load("rul_xgb_improved.pkl")

# Load your test set (from training dataset or wherever you want)
test_df = pd.read_csv("AzureDataset/training_dataset_with_rul.csv")

# Create test split or just take a chunk for prediction
test_sample = test_df.sample(5, random_state=42)  # or however you like

# Match the feature columns exactly like in training
X_test = test_sample.drop(columns=['RUL_hours', 'machineid'])  # keep 'failure_in_24h'

# Predict
dtest = xgb.DMatrix(X_test)
predictions = model.predict(dtest)

# Show result
for i, pred in enumerate(predictions):
    print(f"Sample {i+1} - Predicted RUL: {pred:.2f} hours")

actual_rul = test_sample['RUL_hours'].values
for i, (pred, actual) in enumerate(zip(predictions, actual_rul)):
    print(f"Sample {i+1} - Predicted RUL: {pred:.2f} hours | Actual RUL: {actual} hours")

import matplotlib.pyplot as plt

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
