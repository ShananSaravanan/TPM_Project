import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the saved model
model = joblib.load('xgboost_rul_spc_model.pkl')

# Load test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')['RUL']

# Extract datetime and machineID for output
test_metadata = X_test[['datetime', 'machineid']]

# Ensure feature names match the training set
model_features = model.get_booster().feature_names
X_test_features = X_test[model_features]

# Predict on test set (log scale)
y_pred_log = model.predict(X_test_features)

# Convert back to original scale
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

# Calculate metrics on original scale
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
mae = mean_absolute_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)

print(f"Evaluation Metrics (Original Scale):")
print(f"RMSE: {rmse:.2f} hours")
print(f"MAE: {mae:.2f} hours")
print(f"R²: {r2:.2f}")

# Create actual vs predicted DataFrame and save to CSV
actual_vs_predicted = pd.DataFrame({
    'datetime': test_metadata['datetime'],
    'machineID': test_metadata['machineid'],
    'Actual_RUL': y_test_orig,
    'Predicted_RUL': y_pred
})
actual_vs_predicted.to_csv('actual_vs_predicted.csv', index=False)
print("✅ Saved 'actual_vs_predicted.csv' with datetime, machineID, actual and predicted RUL values (original scale).")

# Plot predicted vs actual RUL (original scale)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, y_pred, alpha=0.5, color='blue')
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
plt.xlabel('Actual RUL (hours)')
plt.ylabel('Predicted RUL (hours)')
plt.title('Predicted vs Actual RUL (Original Scale)')
plt.grid(True)
plt.show()

# Plot residuals
residuals = y_test_orig - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color='green')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted RUL (hours)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot (Original Scale)')
plt.grid(True)
plt.show()

# Detailed feature importance
importance = model.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Feature': list(importance.keys()),
    'Importance': list(importance.values())
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance (Gain):")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance (Gain)')
plt.title('Feature Importance (Gain)')
plt.tight_layout()
plt.show()

# Check for outliers
outliers = np.where((y_pred - y_test_orig).abs() > rmse * 2)[0]
if len(outliers) > 0:
    print(f"\nNumber of significant outliers (error > 2*RMSE): {len(outliers)}")
    print(f"Indices of outliers: {outliers}")
else:
    print("\nNo significant outliers detected (error > 2*RMSE).")