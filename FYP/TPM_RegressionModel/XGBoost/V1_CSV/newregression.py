import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('FYP/TPM_RegressionModel/AzureDataset/training_dataset_with_rul.csv')  # replace with your actual path if needed

# Features and target
features = ['volt', 'rotate', 'pressure', 'vibration']
X = df[features]
y = df['RUL_hours']

# Log-transform the target to handle skew
y_log = np.log1p(y)  # log(1 + RUL)

# Train/test split
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train_log)

# Predict in log-space and convert back
y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)        # back to normal RUL
y_test = np.expm1(y_test_log)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"🔍 Evaluation Metrics:")
print(f"MAE:  {mae:.2f} hours")
print(f"RMSE: {rmse:.2f} hours")

# Plot: Actual vs Predicted RUL
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.4, label='Predictions', color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction')

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted RUL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Save predictions for inspection
output_df = X_test.copy()
output_df['Actual_RUL'] = y_test
output_df['Predicted_RUL'] = y_pred
output_df.to_excel("rul_predictions_output.xlsx", index=False)
print("✅ Predictions saved to 'rul_predictions_regression2.xlsx'.")
