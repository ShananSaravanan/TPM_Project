# test_data.py
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('machine_fault_model.pkl')
			

# Test data
test_data = {
    'temperature': 148.92336873207364,
    'pressure': 22.214121367591716,
    'vibration': 0.10981888071321333,
    'humidity': 39.944953294698905,
    'equipment': 'Turbine',
    'location': 'San Francisco'
}

# Encode categorical variables using the same LabelEncoder as during training
equipment_encoder = LabelEncoder()
location_encoder = LabelEncoder()

# These should be the categories used during training
equipment_encoder.fit(['Turbine', 'Pump', 'Compressor'])  # Fit with all possible values
location_encoder.fit(['Atlanta', 'Chicago', 'New York','San Francisco'])  # Fit with all possible values

# Encode the categorical features
equipment_encoded = equipment_encoder.transform([test_data['equipment']])[0]
location_encoded = location_encoder.transform([test_data['location']])[0]

# Prepare the test data for prediction
processed_data = pd.DataFrame({
    'temperature': [test_data['temperature']],
    'pressure': [test_data['pressure']],
    'vibration': [test_data['vibration']],
    'humidity': [test_data['humidity']],
    'equipment': [equipment_encoded],
    'location': [location_encoded]
})

# Make a prediction with the trained model
prediction = model.predict(processed_data)

# Output the prediction
print(f"Test Data: {test_data}")
print(f"Prediction (0 = Non-Faulty, 1 = Faulty): {prediction[0]}")
