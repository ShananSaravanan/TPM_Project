import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

# Load dataset
df = pd.read_csv('AzureDataset/training_dataset_with_rul.csv')

# Clip RUL and apply log
df['RUL_hours'] = df['RUL_hours'].clip(upper=500)
df['log_rul'] = np.log1p(df['RUL_hours'])

# Optional: Visualize distribution
plt.hist(df['log_rul'], bins=50)
plt.title("Log-Transformed RUL Distribution")
plt.xlabel("log(1 + RUL)")
plt.ylabel("Frequency")
plt.show()

# Prepare data
X = df.drop(columns=['RUL_hours', 'log_rul', 'machineid'])
y = df['log_rul']
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define param grid manually
param_grid = {
    'max_depth': [5, 7],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid = list(ParameterGrid(param_grid))

# Track best model
best_score = float('inf')
best_model = None
best_params = None

print("🔍 Starting manual grid search with GPU...")
for params in tqdm(grid):
    model = XGBRegressor(
        objective='reg:squarederror',
        device='cuda',
        predictor='gpu_predictor',  # 👈 GPU-optimized prediction
        tree_method='hist',         # Better compatibility
        random_state=42,
        **params
    )
    
    # 3-fold CV using negative MSE
    scores = cross_val_score(model, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=3)
    rmse_score = (-scores.mean()) ** 0.5
    
    if rmse_score < best_score:
        best_score = rmse_score
        best_model = model
        best_params = params

print(f"\n✅ Best Params: {best_params}")
print(f"✅ Best CV RMSE: {best_score:.2f} hours")

import matplotlib.pyplot as plt
import xgboost as xgb

xgb.plot_importance(best_model, max_num_features=20)
plt.title("Top Feature Importances")
plt.tight_layout()
plt.show()

# Refit best model on full training data
best_model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(best_model, 'rul_regressor_xgb_log_best_model_gpu.pkl')

# Predict
log_preds = best_model.predict(X_test_scaled)
y_pred = np.expm1(log_preds)
y_true = np.expm1(y_test)

# Evaluate
print("\n📊 Regression Performance (Original RUL scale):")
print(f"MAE:  {mean_absolute_error(y_true, y_pred):.2f} hours")
print(f"MSE:  {mean_squared_error(y_true, y_pred):.2f}")
print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):.2f} hours")
print(f"R²:   {r2_score(y_true, y_pred):.2f}")
