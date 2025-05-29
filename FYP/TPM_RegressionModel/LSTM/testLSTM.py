import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load saved model and scaler
model = load_model("LSTM/Models/lstm_model_v2.h5")
scaler = joblib.load("minmax_scaler.pkl")

# ---------- Load New Data (At least 10 rows) ----------
new_df = pd.read_csv("AzureDataset/test_set.csv")  # <- Replace with your new file

# ---------- Feature Engineering ----------
new_df['pressure_vibration'] = new_df['pressure'] * new_df['vibration']
new_df['rotate_vibration'] = new_df['rotate'] * new_df['vibration']
new_df['volt_rotate'] = new_df['volt'] * new_df['rotate']

features = ['volt', 'rotate', 'pressure', 'vibration',
            'pressure_vibration', 'rotate_vibration', 'volt_rotate']

# Check if enough data
if len(new_df) < 10:
    raise ValueError("❌ At least 10 rows of telemetry data are required for LSTM prediction.")

# ---------- Scale and Reshape Input ----------
X_scaled = scaler.transform(new_df[features])
sequence_input = X_scaled[-10:].reshape(1, 10, len(features))  # shape: (1, 10, 7)

# ---------- Predict ----------
predicted_rul = model.predict(sequence_input).flatten()[0]

# ---------- Display Output ----------
print(f"✅ Predicted Remaining Useful Life (RUL): {predicted_rul:.2f} hours")
