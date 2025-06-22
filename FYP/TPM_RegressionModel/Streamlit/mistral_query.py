# mistral_query.py
import requests
import pandas as pd
from sqlalchemy.sql import text
import re
from datetime import timedelta

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
    
    # Machine details
    query = text("SELECT * FROM machines WHERE machineid = :machine_id")
    machine = pd.read_sql(query, engine, params={"machine_id": machine_id})
    if not machine.empty:
        row = machine.iloc[0]
        # Use age directly as it is stored in years
        age_years = row['age']
        context_parts.append(f"- Model: {row['model']}, Age: {age_years:.2f} years")

    # Failure details
    query = text("SELECT * FROM failures WHERE machineid = :machine_id")
    failures = pd.read_sql(query, engine, params={"machine_id": machine_id})
    if not failures.empty:
        context_parts.append(f"- Recent Failures: {', '.join(failures['failure'])}")

        # Compute telemetry averages during failure periods
        failure_telemetry = []
        for _, failure in failures.iterrows():
            failure_time = failure['datetime']
            # Define a time window around failure (±1 hour)
            start_time = failure_time - timedelta(hours=1)
            end_time = failure_time + timedelta(hours=1)
            telemetry_query = text("""
                SELECT * FROM telemetry 
                WHERE machineid = :machine_id 
                AND datetime BETWEEN :start_time AND :end_time
            """)
            telemetry = pd.read_sql(telemetry_query, engine, params={
                "machine_id": machine_id,
                "start_time": start_time,
                "end_time": end_time
            })
            if not telemetry.empty:
                failure_telemetry.append(telemetry)
        
        if failure_telemetry:
            failure_telemetry_df = pd.concat(failure_telemetry)
            vib_mean = failure_telemetry_df['vibration'].mean()
            pres_mean = failure_telemetry_df['pressure'].mean()
            rot_mean = failure_telemetry_df['rotate'].mean()
            volt_mean = failure_telemetry_df['volt'].mean()
            context_parts.append(
                f"- Telemetry Averages During Failures: Vibration={vib_mean:.2f}, "
                f"Pressure={pres_mean:.2f}, Voltage={volt_mean:.2f}, Rotate={rot_mean:.2f}"
            )

    # Error details
    query = text("SELECT * FROM errors WHERE machineid = :machine_id")
    errors = pd.read_sql(query, engine, params={"machine_id": machine_id}) 
    if not errors.empty:
        context_parts.append(f"- Errors: {', '.join(errors['errorid'])}")

    # Maintenance details
    query = text("SELECT * FROM maintenance WHERE machineid = :machine_id")
    maintenance = pd.read_sql(query, engine, params={"machine_id": machine_id}) 
    if not maintenance.empty:
        latest = maintenance["datetime"].max()
        comps = ', '.join(maintenance['comp'].unique())
        context_parts.append(f"- Last Maintenance: {latest}, Components: {comps}")

    # General telemetry (last 100 readings)
    query = text("SELECT * FROM telemetry WHERE machineid = :machine_id ORDER BY datetime DESC LIMIT 100")
    telemetry = pd.read_sql(query, engine, params={"machine_id": machine_id})
    if not telemetry.empty:
        vib_mean = telemetry['vibration'].mean()
        pres_mean = telemetry['pressure'].mean()
        volt_mean = telemetry['volt'].mean()
        rot_mean = telemetry['rotate'].mean()
        latest = telemetry.iloc[0]
        context_parts.append(
            f"- Recent Telemetry Averages (Last 100 Readings): Vibration={vib_mean:.2f}, "
            f"Pressure={pres_mean:.2f}, Voltage={volt_mean:.2f}, Rotate={rot_mean:.2f}\n"
            f"- Latest Reading: Vibration={latest['vibration']}, Pressure={latest['pressure']}, "
            f"Voltage={latest['volt']}, Rotate={latest['rotate']} at {latest['datetime']}"
        )

    return '\n'.join(context_parts)

def get_global_context(engine, question):
    context_parts = ["Global Context:\n"]
    
    # Define keyword-to-context mappings
    keyword_context = {
        'age|oldest|older': {
            'query': text("SELECT machineid, model, age FROM machines ORDER BY age DESC LIMIT 5"),
            'format': lambda df: '\n'.join([f"- Machine {row.machineid}: Model={row.model}, Age={row.age:.2f} years" for _, row in df.iterrows()]),
            'label': "Oldest Machines"
        },
        'failure|failed|fails': {
            'query': text("SELECT machineid, COUNT(*) AS fail_count FROM failures GROUP BY machineid ORDER BY fail_count DESC LIMIT 5"),
            'format': lambda df: '\n'.join([f"- Machine {row.machineid}: {row.fail_count} failures" for _, row in df.iterrows()]),
            'label': "Machines with Highest Failure Counts"
        },
        'rul|remaining life|useful life': {
            'query': text("SELECT * FROM predictions ORDER BY rul_pred ASC LIMIT 5"),
            'format': lambda df: '\n'.join([f"- Machine {row.machineid}: RUL={row.rul_pred:.2f} at {row.prediction_time}" for _, row in df.iterrows()]),
            'label': "Machines with Lowest RUL"
        },
        'telemetry|sensor|vibration|rotate|pressure|volt': {
            'query': text("SELECT machineid, AVG(vibration) AS vib_mean, AVG(pressure) AS pres_mean, AVG(rotate) AS rot_mean, AVG(volt) AS volt_mean FROM telemetry GROUP BY machineid ORDER BY machineid LIMIT 5"),
            'format': lambda df: '\n'.join([f"- Machine {row.machineid}: Vibration={row.vib_mean:.2f}, Pressure={row.pres_mean:.2f}, Rotate={row.rot_mean:.2f}, Voltage={row.volt_mean:.2f}" for _, row in df.iterrows()]),
            'label': "Telemetry Averages Across Machines"
        }
    }
    
    # Convert question to lowercase for matching
    question_lower = question.lower()
    
    # Check which keywords are present in the question
    matched_keywords = []
    for pattern in keyword_context:
        if re.search(pattern, question_lower):
            matched_keywords.append(pattern)
    
    # If no keywords are matched, include default context (age, RUL, failures)
    if not matched_keywords:
        matched_keywords = ['age|oldest|older', 'rul|remaining life|useful life', 'failure|failed|fails']
    
    # Fetch and format context for matched keywords
    for pattern in matched_keywords:
        context_info = keyword_context[pattern]
        df = pd.read_sql(context_info['query'], engine)
        if not df.empty:
            context_parts.append(f"{context_info['label']}:\n{context_info['format'](df)}")

    # If no data was added (e.g., all tables empty), provide a fallback message
    if len(context_parts) == 1:
        context_parts.append("No relevant data available for the query.")

    return '\n'.join(context_parts)

def build_prompt(machine_id_or_none, question, engine):
    base_intro = (
        "You are a predictive maintenance AI assistant with access to machine telemetry, errors, failures, "
        "maintenance logs, and RUL predictions. Answer user questions using available context. Be precise and explain clearly.\n"
    )
    
    if machine_id_or_none:
        context = get_machine_context(engine, machine_id_or_none)
    else:
        context = get_global_context(engine, question)

    return f"{base_intro}{context}\nUser question: {question}\nAnswer:"