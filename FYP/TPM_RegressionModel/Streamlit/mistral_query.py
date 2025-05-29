# mistral_query.py
import requests
import pandas as pd
from sqlalchemy.sql import text
import re

def query_mistral(prompt):
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
        "temperature": 0.5
    })
    return res.json()["response"]

def get_machine_context(engine, machine_id):
    context_parts = [f"Context for Machine ID: {machine_id}\n"]
    
    query = text("SELECT * FROM machines WHERE machineid = :machine_id")
    machine = pd.read_sql(query, engine, params={"machine_id": machine_id})
    if not machine.empty:
        row = machine.iloc[0]
        context_parts.append(f"- Model: {row['model']}, Age: {row['age']} days")

    query = text("SELECT * FROM failures WHERE machineid = :machine_id")
    failures = pd.read_sql(query, engine, params={"machine_id": machine_id})
    if not failures.empty:
        context_parts.append(f"- Recent Failures: {', '.join(failures['failure'])}")

    query = text("SELECT * FROM errors WHERE machineid = :machine_id")
    errors = pd.read_sql(query, engine, params={"machine_id": machine_id}) 
    if not errors.empty:
        context_parts.append(f"- Errors: {', '.join(errors['errorid'])}")

    query = text("SELECT * FROM maintenance WHERE machineid = :machine_id")
    maintenance = pd.read_sql(query, engine, params={"machine_id": machine_id}) 
    if not maintenance.empty:
        latest = maintenance["datetime"].max()
        comps = ', '.join(maintenance['comp'].unique())
        context_parts.append(f"- Last Maintenance: {latest}, Components: {comps}")

    query = text("SELECT * FROM telemetry WHERE machineid = :machine_id ORDER BY datetime DESC LIMIT 100")
    telemetry = pd.read_sql(query, engine, params={"machine_id": machine_id})
    if not telemetry.empty:
        vib_mean = telemetry['vibration'].mean()
        pres_mean = telemetry['pressure'].mean()
        volt_mean = telemetry['volt'].mean()
        latest = telemetry.iloc[0]
        context_parts.append(
            f"- Telemetry Averages: Vibration={vib_mean:.2f}, Pressure={pres_mean:.2f}, Voltage={volt_mean:.2f}\n"
            f"- Latest Reading: V={latest['vibration']}, P={latest['pressure']}, U={latest['volt']} at {latest['datetime']}"
        )

    return '\n'.join(context_parts)

def get_global_context(engine):
    context_parts = ["Global Context:\n"]

    # Machines with lowest RUL
    rul_query = text("SELECT * FROM predictions ORDER BY rul_pred ASC LIMIT 5")
    rul_df = pd.read_sql(rul_query, engine)
    if not rul_df.empty:
        rul_info = '\n'.join([f"- Machine {row.machineid}: RUL={row.rul_pred:.2f} at {row.prediction_time}" for _, row in rul_df.iterrows()])
        context_parts.append("Machines with lowest RUL:\n" + rul_info)

    # Failure counts
    fail_count_query = text("SELECT machineid, COUNT(*) AS fail_count FROM failures GROUP BY machineid ORDER BY fail_count DESC LIMIT 5")
    fail_df = pd.read_sql(fail_count_query, engine)
    if not fail_df.empty:
        fail_info = '\n'.join([f"- Machine {row.machineid}: {row.fail_count} failures" for _, row in fail_df.iterrows()])
        context_parts.append("Machines with highest failure counts:\n" + fail_info)

    return '\n'.join(context_parts)

def build_prompt(machine_id_or_none, question, engine):
    base_intro = (
        "You are a predictive maintenance AI assistant with access to machine telemetry, errors, failures, "
        "maintenance logs, and RUL predictions. Answer user questions using available context. Be precise and explain clearly.\n"
    )
    
    if machine_id_or_none:
        context = get_machine_context(engine, machine_id_or_none)
    else:
        context = get_global_context(engine)

    return f"{base_intro}{context}\nUser question: {question}\nAnswer:"
