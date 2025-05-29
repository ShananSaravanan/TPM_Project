# rul_predictor.py
import joblib
import pandas as pd

model = joblib.load("XGBoost/Models/xgboost_rul_model(postgre).pkl")

def predict_rul(data: pd.DataFrame) -> pd.DataFrame:
    features = ['volt', 'rotate', 'pressure', 'vibration', 'age', 'volt_rotate', 'pressure_vibration']
    data['volt_rotate'] = data['volt'] * data['rotate']
    data['pressure_vibration'] = data['pressure'] * data['vibration']
    data['rul_pred'] = model.predict(data[features])
    return data
