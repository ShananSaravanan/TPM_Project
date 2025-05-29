from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import joblib

# Load dataset
df = pd.read_csv("TPM_RegressionModel/AzureDataset/training_dataset.csv")

# # Optional: Sample the dataset (0.5% in your original code)
# df_sampled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and label
X = df.drop(['failure_in_24h', 'machineid'], axis=1)
y = df['failure_in_24h']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GPU-enabled XGBoost model
model = XGBClassifier(
    tree_method='gpu_hist',         # Enables GPU training
    predictor='gpu_predictor',      # Uses GPU for prediction too
    n_estimators=100,
    random_state=42,
    verbosity=1,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Train model
print("🚀 Training XGBoost on GPU...")
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'azuretpm_model_xgboost.pkl')
