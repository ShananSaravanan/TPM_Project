# mock_sender.py
import requests
import random
import time
from datetime import datetime

# Machine IDs to simulate
machine_ids = list(range(1, 101))  # includes 1 to 100

while True:
    for machine_id in machine_ids:
        data = {
            "machineid": machine_id,
            "datetime": datetime.now().isoformat(),
            "volt": round(random.uniform(30.0, 35.0), 2),
            "rotate": round(random.uniform(1300, 1500), 2),
            "pressure": round(random.uniform(100, 120), 2),
            "vibration": round(random.uniform(30, 40), 2)
        }

        try:
            response = requests.post("http://localhost:8000/telemetry", json=data)
            if response.status_code == 200:
                print(f"[{machine_id}] ✅ Sent: {data['datetime']} | Status: {response.status_code}")
            else:
                print(f"[{machine_id}] ❌ Failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[{machine_id}] ❌ Failed to send: {e}")

    time.sleep(5)  # Send every 5 seconds