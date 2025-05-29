import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# ===== Load Dataset =====
df = pd.read_csv('AzureDataset/training_dataset_with_rul.csv')
df['RUL_hours'] = df['RUL_hours']

# Features
top_features = [
    'volt_mean_3h', 'rotate_mean_3h', 'pressure_mean_3h', 'vibration_mean_3h',
    'volt_mean_24h', 'rotate_mean_24h', 'pressure_mean_24h', 'vibration_mean_24h',
    'volt', 'rotate', 'pressure', 'vibration', 'age'
]
X = df[top_features]
y = df['RUL_hours']

# ===== Shuffle + Split =====
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== Scaling =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== Baseline: Linear Regression =====
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)

print("\n📊 Linear Regression Performance:")
print(f"MAE:  {mean_absolute_error(y_test, lr_preds):.2f} hours")
print(f"MSE:  {mean_squared_error(y_test, lr_preds):.2f}")
print(f"RMSE: {mean_squared_error(y_test, lr_preds, squared=False):.2f} hours")
print(f"R²:   {r2_score(y_test, lr_preds):.2f}")

# ===== XGBoost Model =====
xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    tree_method='hist',
    device='cuda',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6
)
xgb.fit(X_train_scaled, y_train)
xgb_preds = xgb.predict(X_test_scaled)
import joblib
joblib.dump(xgb, "xgb_rul_model.pkl")
print("\n📊 XGBoost Performance:")
print(f"MAE:  {mean_absolute_error(y_test, xgb_preds):.2f} hours")
print(f"MSE:  {mean_squared_error(y_test, xgb_preds):.2f}")
print(f"RMSE: {mean_squared_error(y_test, xgb_preds, squared=False):.2f} hours")
print(f"R²:   {r2_score(y_test, xgb_preds):.2f}")

# ===== Visualizations =====
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot([0, 500], [0, 500], 'r--')
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_predictions(y_test, lr_preds, "Linear Regression: True vs Predicted RUL")
plot_predictions(y_test, xgb_preds, "XGBoost: True vs Predicted RUL")

# ===== Residuals =====
residuals = y_test - xgb_preds
plt.figure(figsize=(8, 4))
plt.hist(residuals, bins=50, color='salmon', edgecolor='black')
plt.title("XGBoost Residuals (True - Predicted RUL)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# ===== Feature Importance =====
plot_importance(xgb, max_num_features=10)
plt.title("Top Feature Importances (XGBoost)")
plt.show()
