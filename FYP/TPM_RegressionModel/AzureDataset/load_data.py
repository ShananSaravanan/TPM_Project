import pandas as pd
import psycopg2

# Connect to DB
conn = psycopg2.connect(
    dbname="AzureTPMDB",
    user="postgres",
    password="root",
    host="localhost",
    port="5433"
)

# Load telemetry data
telemetry = pd.read_sql("SELECT * FROM telemetry ORDER BY machineid, datetime", conn)
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])

# Load machines and merge
machines = pd.read_sql("SELECT * FROM machines", conn)
telemetry = telemetry.merge(machines, on='machineid', how='left')

# Add rolling features
def add_rolling_features(df, cols, windows=[3, 24]):
    for col in cols:
        for w in windows:
            df[f'{col}_mean_{w}h'] = df.groupby('machineid')[col].transform(
                lambda x: x.rolling(window=w, min_periods=1).mean()
            )
    return df

telemetry = add_rolling_features(telemetry, ['volt', 'rotate', 'pressure', 'vibration'])

# Load failures
failures = pd.read_sql("SELECT * FROM failures", conn)
failures['datetime'] = pd.to_datetime(failures['datetime'])

# Label future failures
telemetry['failure_in_24h'] = 0
for _, row in failures.iterrows():
    affected_machine = row['machineid']
    fail_time = row['datetime']
    mask = (
        (telemetry['machineid'] == affected_machine) &
        (telemetry['datetime'] >= fail_time - pd.Timedelta(hours=24)) &
        (telemetry['datetime'] < fail_time)
    )
    telemetry.loc[mask, 'failure_in_24h'] = 1

# One-hot encode model
telemetry = pd.get_dummies(telemetry, columns=['model'], drop_first=True)

# Preview
print("Final shape:", telemetry.shape)
print(telemetry.head())

# Optional: Save for ML use
features = telemetry.drop(columns=['datetime'])
features.to_csv('training_dataset.csv', index=False)

# Make a copy to avoid modifying original
telemetry_rul = telemetry.copy()

# Ensure datetime is sorted
telemetry_rul = telemetry_rul.sort_values(by=['machineid', 'datetime'])

# Add a column to hold RUL in hours
telemetry_rul['RUL_hours'] = None

# For each machine, calculate time to next failure
for machine_id in telemetry_rul['machineid'].unique():
    machine_data = telemetry_rul[telemetry_rul['machineid'] == machine_id]
    machine_failures = failures[failures['machineid'] == machine_id].sort_values('datetime')
    
    for idx, row in machine_data.iterrows():
        future_failures = machine_failures[machine_failures['datetime'] > row['datetime']]
        if not future_failures.empty:
            time_to_fail = (future_failures.iloc[0]['datetime'] - row['datetime']).total_seconds() / 3600.0
            telemetry_rul.at[idx, 'RUL_hours'] = time_to_fail
        else:
            telemetry_rul.at[idx, 'RUL_hours'] = None  # No future failure

# Drop rows where RUL is NaN (we can’t train on those)
telemetry_rul = telemetry_rul.dropna(subset=['RUL_hours'])

# Optional: Save
telemetry_rul.drop(columns=['datetime']).to_csv('training_dataset_with_rul.csv', index=False)

print(telemetry_rul[['machineid', 'datetime', 'RUL_hours']].head(10))
