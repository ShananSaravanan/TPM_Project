import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load test data and model
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')['RUL']
model = joblib.load('xgboost_rul_model.pkl')

# Define features in the exact order as training
features = ['rotate', 
            'rotate_rolling_mean_6', 'rotate_rolling_mean_12', 'rotate_rolling_mean_24',
            'rotate_rolling_std_6', 'rotate_rolling_std_12', 'rotate_rolling_std_24',
            'rotate_ema_6', 'rotate_change', 'rotate_outlier']

# Verify features exist in X_test
missing_features = [f for f in features if f not in X_test.columns]
if missing_features:
    raise ValueError(f"Missing features in X_test: {missing_features}")

# Prepare test features
X_test_features = X_test[features]

# Make predictions (sqrt scale)
y_pred = model.predict(X_test_features)
y_pred = np.maximum(y_pred, 0)  # Ensure non-negative predictions

# Evaluate on sqrt scale
rmse_sqrt = np.sqrt(mean_squared_error(y_test, y_pred))
mae_sqrt = mean_absolute_error(y_test, y_pred)
r2_sqrt = r2_score(y_test, y_pred)
print("Evaluation on Square-Root Scale:")
print(f"RMSE (sqrt hours): {rmse_sqrt:.2f}")
print(f"MAE (sqrt hours): {mae_sqrt:.2f}")
print(f"R²: {r2_sqrt:.2f}")

# Reverse square-root transformation for original scale (hours)
y_test_orig = y_test ** 2
y_pred_orig = y_pred ** 2

# Evaluate on original scale
rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
r2_orig = r2_score(y_test_orig, y_pred_orig)
print("\nEvaluation on Original Scale (hours):")
print(f"RMSE: {rmse_orig:.2f} hours")
print(f"MAE: {mae_orig:.2f} hours")
print(f"R²: {r2_orig:.2f}")

# Plot actual vs. predicted RUL (original scale)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, y_pred_orig, alpha=0.5, s=20)
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', label='Ideal')
plt.xlabel('Actual RUL (hours)')
plt.ylabel('Predicted RUL (hours)')
plt.title('Actual vs. Predicted RUL')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot residuals (original scale)
residuals = y_test_orig - y_pred_orig
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_orig, residuals, alpha=0.5, s=20)
plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
plt.xlabel('Predicted RUL (hours)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Summary statistics of predictions
print("\nPredicted RUL (hours) Summary:")
print(pd.Series(y_pred_orig).describe())
print("\nActual RUL (hours) Summary:")
print(pd.Series(y_test_orig).describe())