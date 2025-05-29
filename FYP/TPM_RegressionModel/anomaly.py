from sklearn.ensemble import IsolationForest
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.utils import shuffle
from xgboost import callback

# Load the dataset
df = pd.read_csv('TPM_RegressionModel/AzureDataset/training_dataset_with_rul.csv')

# Features for anomaly detection (selecting columns from the dataset)
X_telemetry = df[['volt', 'rotate', 'pressure', 'vibration', 'volt_mean_3h', 'volt_mean_24h', 
                  'rotate_mean_3h', 'rotate_mean_24h', 'pressure_mean_3h', 'pressure_mean_24h', 
                  'vibration_mean_3h', 'vibration_mean_24h']]

# Initialize the XGBoost anomaly model (GPU-enabled)
params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',  # Use 'hist' method
    'device': 'cuda',       # Specify CUDA device for GPU support
    'eval_metric': 'logloss',
    'learning_rate': 0.05,
    'max_depth': 7,
    'random_state': 42,
    'base_score': 0.5
}


# Convert data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_telemetry)

# Set up the progress bar using tqdm for tracking the training progress
num_round = 100

class TQDMCallback(callback.TrainingCallback):
    def __init__(self, total_rounds):
        self.pbar = tqdm(total=total_rounds, desc="Anomaly Detection Training Progress", ncols=100)

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model  # Make sure to return the model here

# Train the XGBoost model with the callback to update the progress bar
bst = xgb.train(params, dtrain, num_boost_round=num_round, callbacks=[TQDMCallback(total_rounds=num_round)])

# Make predictions (anomalies: 1 = normal, 0 = anomaly)
df['anomaly_score'] = bst.predict(dtrain)  # Anomaly score: higher = more normal
df['anomaly'] = (df['anomaly_score'] < 0.5).astype(int)  # Anomaly: 1 = normal, 0 = anomaly (threshold at 0.5)

# Show results
print(df[['machineid', 'anomaly_score', 'anomaly']].head())

# Save the results with anomalies to a new CSV
df.to_csv('training_dataset_with_anomalies.csv', index=False)
