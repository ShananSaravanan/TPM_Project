import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import stats

# Load the saved model
model = joblib.load('FYP/TPM_RegressionModel/XGBoost/V2_Postgre/Model/xgboost_rul_spc_model.pkl')

# Load test data
X_test = pd.read_csv('FYP/TPM_RegressionModel/XGBoost/V2_Postgre/X_test.csv')
y_test_sqrt = pd.read_csv('FYP/TPM_RegressionModel/XGBoost/V2_Postgre/y_test.csv')['RUL']  # Loaded as square root scale

# Debug: Print columns and RUL range
print("X_test columns:", X_test.columns.tolist())
print("y_test_orig min:", (y_test_sqrt ** 2).min(), "y_test_orig max:", (y_test_sqrt ** 2).max())

# Ensure feature names match the training set
model_features = model.get_booster().feature_names
X_test = X_test[model_features].fillna(0)  # Handle any missing values

# Predict on test set (square root scale)
y_pred_sqrt = model.predict(X_test)

# Inverse transform to original scale
y_pred_orig = y_pred_sqrt ** 2  # Reverse square root
y_test_orig = y_test_sqrt ** 2  # Reverse square root to match

# Calculate regression metrics on original scale
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mae = mean_absolute_error(y_test_orig, y_pred_orig)
r2 = r2_score(y_test_orig, y_pred_orig)
mask = y_test_orig > 1  # Exclude values <= 1 hour to avoid MAPE issues
mape = mean_absolute_percentage_error(y_test_orig[mask], y_pred_orig[mask]) * 100 if mask.any() else np.nan

print(f"Evaluation Metrics (Original Scale):")
print(f"RMSE: {rmse:.2f} hours")
print(f"MAE: {mae:.2f} hours")
print(f"R²: {r2:.2f}")
print(f"MAPE: {mape:.2f}%")

# Threshold for classification (e.g., median RUL on original scale)
threshold = np.median(y_test_orig)  # Use median of original scale
y_test_binary = (y_test_orig <= threshold).astype(int)
y_pred_binary = (y_pred_orig <= threshold).astype(int)

# Calculate classification metrics
precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
recall = recall_score(y_test_binary, y_pred_binary)
f1 = f1_score(y_test_binary, y_pred_binary)

print(f"Classification Metrics (Threshold = {threshold:.0f} hours):")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Create actual vs predicted DataFrame and save to CSV
actual_vs_predicted = pd.DataFrame({
    'datetime': X_test['datetime'],
    'machineID': X_test['machineID'],
    'Actual_RUL': y_test_orig,
    'Predicted_RUL': y_pred_orig
})
actual_vs_predicted.to_csv('actual_vs_predicted.csv', index=False)
print("✅ Saved 'actual_vs_predicted.csv' with datetime, machineID, actual and predicted RUL values (original scale).")

# Plot 1: Predicted vs Actual RUL (original scale)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, y_pred_orig, alpha=0.5, color='blue')
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
plt.xlabel('Actual RUL (hours)')
plt.ylabel('Predicted RUL (hours)')
plt.title('Predicted vs Actual RUL (Original Scale)')
plt.grid(True)
plt.show()

# Plot 2: Residuals (original scale)
residuals = y_test_orig - y_pred_orig
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_orig, residuals, alpha=0.5, color='green')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted RUL (hours)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot (Original Scale)')
plt.grid(True)
plt.show()

# Plot 3: Residual Histogram
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=50, color='purple', edgecolor='black')
plt.xlabel('Residuals (hours)')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.grid(True)
plt.show()

# Plot 4: Prediction Interval (95% Confidence Interval)
confidence = 0.95
squared_errors = (y_test_orig - y_pred_orig) ** 2
mean_se = np.mean(squared_errors)
std_se = np.std(squared_errors)
ci = np.sqrt(mean_se + std_se * stats.t.ppf((1 + confidence) / 2, len(y_test_orig) - 1))
plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, y_pred_orig, alpha=0.5, color='blue')
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
plt.fill_between([y_test_orig.min(), y_test_orig.max()], 
                 [y_test_orig.min() - ci, y_test_orig.max() - ci], 
                 [y_test_orig.min() + ci, y_test_orig.max() + ci], 
                 color='gray', alpha=0.2, label='95% CI')
plt.xlabel('Actual RUL (hours)')
plt.ylabel('Predicted RUL (hours)')
plt.title('Predicted vs Actual RUL with 95% Confidence Interval')
plt.legend()
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

# Plot 5: Feature Importance
plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance (Gain)')
plt.title('Feature Importance (Gain)')
plt.tight_layout()
plt.show()

# Check for outliers with adjustable threshold
outlier_threshold = 100  # Adjusted to 100 hours for practical significance
outliers = np.where((y_pred_orig - y_test_orig).abs() > outlier_threshold)[0]
if len(outliers) > 0:
    print(f"\nNumber of significant outliers (error > {outlier_threshold} hours): {len(outliers)}")
    print(f"Indices of outliers: {outliers}")
    # Optional: Print outlier details
    outliers_df = X_test.iloc[outliers][['datetime', 'machineID']]
    print("Outlier Details (datetime, machineID):")
    print(outliers_df.head())
else:
    print(f"\nNo significant outliers detected (error > {outlier_threshold} hours).")