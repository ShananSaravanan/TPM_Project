import pandas as pd
import numpy as np

# Load CSVs (without errors='coerce')
telemetry = pd.read_csv("AzureDataset/PdM_telemetry.csv")
failures = pd.read_csv("AzureDataset/PdM_failures.csv")

# Convert to datetime explicitly with error handling
telemetry["datetime"] = pd.to_datetime(telemetry["datetime"], dayfirst=True, errors='coerce')
failures["datetime"] = pd.to_datetime(failures["datetime"], dayfirst=True, errors='coerce')

# Drop invalid datetimes if any
telemetry = telemetry.dropna(subset=["datetime"])
failures = failures.dropna(subset=["datetime"])

# Sort
telemetry.sort_values(by=["machineID", "datetime"], inplace=True)
failures.sort_values(by=["machineID", "datetime"], inplace=True)

# Compute RUL
def compute_rul(row, failures_df):
    future_failures = failures_df[
        (failures_df["machineID"] == row["machineID"]) & 
        (failures_df["datetime"] > row["datetime"])
    ]
    if not future_failures.empty:
        next_failure_time = future_failures["datetime"].min()
        return (next_failure_time - row["datetime"]).total_seconds() / 3600  # in hours
    else:
        return np.nan

telemetry["RUL_hours"] = telemetry.apply(lambda row: compute_rul(row, failures), axis=1)
telemetry["log_RUL_hours"] = telemetry["RUL_hours"].apply(lambda x: np.log1p(x) if pd.notnull(x) else np.nan)

# Drop rows with NaN in 'RUL_hours' and 'log_RUL_hours'
telemetry = telemetry.dropna(subset=['RUL_hours', 'log_RUL_hours'])


# Save result
telemetry.to_csv("telemetry_with_rul.csv", index=False)
print(telemetry[["datetime", "machineID", "RUL_hours", "log_RUL_hours"]].head())
