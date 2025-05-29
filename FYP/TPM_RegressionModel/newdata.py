import pandas as pd

# --- Step 1: Load your data (assuming you already have it) ---
telemetry = pd.read_csv('AzureDataset/PdM_telemetry.csv')
failures = pd.read_csv('AzureDataset/PdM_failures.csv')

# For now I assume you already have `telemetry` and `failures` DataFrames from what you pasted.

import pandas as pd

# --- Assume you already have telemetry and failures loaded ---

# Step 1: Make sure datetime is proper type
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
failures['datetime'] = pd.to_datetime(failures['datetime'])

# Step 2: Initialize failure_label if not already present
if 'failure_label' not in telemetry.columns:
    telemetry['failure_label'] = 0

# Step 3: Label data 48h before each failure
for idx, row in failures.iterrows():
    machine_id = row['machineID']
    failure_time = row['datetime']
    
    # Mark telemetry data between [failure_time - 48h, failure_time)
    mask = (
        (telemetry['machineID'] == machine_id) &
        (telemetry['datetime'] >= failure_time - pd.Timedelta(hours=48)) &
        (telemetry['datetime'] < failure_time)
    )
    telemetry.loc[mask, 'failure_label'] = 1



# Save the updated telemetry DataFrame to a new CSV file
telemetry.to_csv('AzureDataset/telemetry_with_labels.csv', index=False)

