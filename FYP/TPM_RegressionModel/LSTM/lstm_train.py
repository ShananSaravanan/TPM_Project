import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load dataset
df = pd.read_csv('AzureDataset/telemetry_with_logrule.csv')

# ---------- Feature Engineering ----------
df['pressure_vibration'] = df['pressure'] * df['vibration']
df['rotate_vibration'] = df['rotate'] * df['vibration']
df['volt_rotate'] = df['volt'] * df['rotate']

# Use more features if needed
features = ['volt', 'rotate', 'pressure', 'vibration',
            'pressure_vibration', 'rotate_vibration', 'volt_rotate']

# Normalize features globally
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])
joblib.dump(scaler, "minmax_scaler.pkl")

# Use log(RUL_hours) instead of raw RUL
y_raw = df['RUL_hours'].values
y_log = np.log1p(y_raw)  # log1p handles log(0) safely

# ---------- Generate Sequences for LSTM ----------
sequence_length = 10
X_seq, y_seq = [], []

for i in range(len(X_scaled) - sequence_length):
    X_seq.append(X_scaled[i:i+sequence_length])
    y_seq.append(y_log[i + sequence_length])  # use log value

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# ---------- Time-Aware Train-Test Split ----------
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# ---------- Build Model ----------
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# ---------- Learning Rate Scheduler ----------
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# ---------- Train Model ----------
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[lr_scheduler]
)

# ---------- Save Model ----------
model.save('LSTM/Models/lstm_model_v2_log.h5')

# ---------- Plot Loss ----------
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# ---------- Evaluate Model ----------
y_pred_log = model.predict(X_test).flatten()
y_pred = np.expm1(y_pred_log)  # Convert back from log space
y_test_exp = np.expm1(y_test)

mae = mean_absolute_error(y_test_exp, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred))
print(f"🔍 MAE: {mae:.2f} hours")
print(f"🔍 RMSE: {rmse:.2f} hours")

# ---------- Plot Predictions ----------
plt.figure(figsize=(10, 6))
plt.scatter(y_test_exp, y_pred, alpha=0.4, label='Predicted vs Actual', color='blue')
plt.plot([y_test_exp.min(), y_test_exp.max()], [y_test_exp.min(), y_test_exp.max()],
         color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted RUL (LSTM with Log RUL)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Save Full Output ----------
aligned_indices = np.arange(train_size + sequence_length, train_size + sequence_length + len(y_test_exp))
output_df_full = df.iloc[aligned_indices].copy()
output_df_full['Actual_RUL'] = y_test_exp
output_df_full['Predicted_RUL'] = y_pred
output_df_full.to_excel("lstm_rul_predictions_log_output.xlsx", index=False)
print("✅ Full telemetry with predictions saved to 'lstm_rul_predictions_log_output.xlsx'")
