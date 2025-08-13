# telemetry_server.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sqlalchemy import create_engine
import uvicorn
import logging
import os
# Initialize app and logging
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS for frontend (optional but helpful for dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db_url = os.environ.get("db_url")
engine = create_engine(db_url)

# # PostgreSQL connection
# db_config = {
#     'host': 'db.gwvaqtikukibtwmtbpar.supabase.co',
#     'port': '5432',
#     'database': 'postgres',
#     'user': 'postgres',
#     'password': 'BigRoot@123@'
# }
# conn_str = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
# engine = create_engine(conn_str)

@app.post("/telemetry")
async def receive_telemetry(request: Request):
    try:
        data = await request.json()
        logger.info(f"Received data: {data}")
        df = pd.DataFrame([data])
        # Ensure only expected columns are inserted
        expected_columns = ["machineid", "datetime", "volt", "rotate", "pressure", "vibration"]
        df = df[expected_columns]
        df.to_sql("telemetry", engine, if_exists="append", index=False)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing telemetry: {e}")
        return {"status": "error", "message": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run("telemetry_server:app", host="0.0.0.0", port=8000, reload=True)