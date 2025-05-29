import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import time
from model_predictor import predict_rul

engine = create_engine("postgresql://postgres:root@localhost:5433/AzureTPMDB")

while True:
    try:
        telemetry = pd.read_sql("SELECT * FROM telemetry ORDER BY datetime DESC LIMIT 10", engine)
        machines = pd.read_sql("SELECT * FROM machines", engine)
        data = telemetry.merge(machines, on='machineid', how='left')
        predictions_df = predict_rul(data)
        predictions_df['prediction_time'] = datetime.now()

        predictions_df[['machineid', 'prediction_time', 'rul_pred']].to_sql(
            'predictions', engine, if_exists='append', index=False
        )
        print("✅ Live prediction inserted.")
    except Exception as e:
        print("❌ Prediction failed:", e)

    time.sleep(10)  # adjust refresh rate
