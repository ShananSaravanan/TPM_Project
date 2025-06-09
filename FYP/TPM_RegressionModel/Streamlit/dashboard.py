import streamlit as st
import pandas as pd
import time
from sqlalchemy import create_engine
from sqlalchemy.sql import text
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import streamlit.components.v1 as components

# --------------------------- 
# 🎛 Page & DB Setup
# --------------------------- 
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("🛠️ Predictive Maintenance Dashboard")

engine = create_engine("postgresql://postgres:root@localhost:5433/AzureTPMDB")

# --------------------------- 
# 🔧 Machine Selector
# --------------------------- 
machine_ids = pd.read_sql("SELECT DISTINCT machineid FROM predictions", engine)["machineid"].tolist()
selected_machine = st.selectbox("🔧 Select Machine", machine_ids)

# --------------------------- 
# 📊 Live Graph Tabs
# --------------------------- 
tab1, tab2, tab3 = st.tabs(["🔋 Telemetry", "⏳ RUL per Machine", "📉 All Machines RUL Overview"])

# --------------------------- 
# 🔋 Tab 1 - Telemetry Viewer
# --------------------------- 
with tab1:
    st.subheader(f"Telemetry Trend for Machine {selected_machine}")

    # Select telemetry type to view
    telemetry_option = st.selectbox("Select telemetry type", ["volt", "rotate", "pressure", "vibration"])

    # Initial telemetry data
    tele_query = text("SELECT * FROM telemetry WHERE machineid = :machine_id ORDER BY datetime")
    telemetry = pd.read_sql(tele_query, engine, params={"machine_id": selected_machine})
    telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])
    telemetry.set_index("datetime", inplace=True)

    tele_chart = st.line_chart(telemetry[[telemetry_option]].tail(20))

    # Live update loop (use button to start)
    if st.button("▶️ Start Telemetry Stream"):
        for _ in range(100):  # or use a session state-based condition
            time.sleep(10)
            latest_query = text("""
                SELECT * FROM telemetry 
                WHERE machineid = :machine_id 
                ORDER BY datetime DESC LIMIT 1
            """)
            latest = pd.read_sql(latest_query, engine, params={"machine_id": selected_machine})
            latest["datetime"] = pd.to_datetime(latest["datetime"])
            latest.set_index("datetime", inplace=True)
            tele_chart.add_rows(latest[[telemetry_option]])

# --------------------------- 
# ⏳ Tab 2 - RUL Trend per Machine
# --------------------------- 
def send_telegram_alert(chat_id, message_text):
    bot_token = "7688190828:AAHF3QU70K5A8djY_d3RViMH-uA6Nsl6LL0"
    send_message_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    payload = {
        "chat_id": chat_id,
        "text": message_text,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(send_message_url, data=payload)
        if response.status_code == 200:
            print(f"✅ Telegram alert sent to chat {chat_id}")
        else:
            print(f"❌ Telegram alert failed: {response.text}")
    except Exception as e:
        print(f"❌ Exception sending telegram alert: {e}")

def send_email_alert(machine_id, rul_value):
    sender_email = "shananmessi10@gmail.com"
    sender_password = "aswv tqus gstv wnfc"  # Replace with your actual app password
    recipient_email = "dreshya1423@gmail.com"  # Replace with your recipient email

    subject = f"⚠️ Maintenance Alert: Machine {machine_id} RUL Below 200"
    body = f"Warning! The Remaining Useful Life (RUL) of Machine {machine_id} has dropped below 200 hours.\n\nCurrent RUL: {rul_value} hours"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"✅ Email alert sent to {recipient_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

with tab2:
    st.subheader(f"RUL Prediction Trend for Machine {selected_machine}")

    # Cache RUL data to reduce DB queries
    @st.cache_data(ttl=60)  # Cache for 60 seconds
    def get_rul_data(machine_id):
        rul_query = text("SELECT prediction_time, rul_pred FROM predictions WHERE machineid = :machine_id ORDER BY prediction_time")
        rul_data = pd.read_sql(rul_query, engine, params={"machine_id": machine_id})
        rul_data["prediction_time"] = pd.to_datetime(rul_data["prediction_time"])
        rul_data.set_index("prediction_time", inplace=True)
        return rul_data

    rul_data = get_rul_data(selected_machine)
    rul_chart = st.line_chart(rul_data[["rul_pred"]].tail(30))  # Show last 30 RULs

    # Check if RUL goes below threshold and trigger an alert
    latest_rul_value = rul_data["rul_pred"].iloc[-1] if not rul_data.empty else float('inf')
    alert_threshold = 200  # Adjusted threshold based on model context

    if latest_rul_value < alert_threshold:
        st.warning(f"⚠️ Warning! RUL is below {alert_threshold} hours! Maintenance is recommended.", icon="⚠️")
        st.markdown("### 📨 Email Alert Sent for Low RUL!")
        send_email_alert(selected_machine, latest_rul_value)
        chat_id = -1002671447415  # Your Telegram supergroup chat ID
        telegram_message = (
            f"⚠️ <b>Maintenance Alert</b>\n"
            f"Machine {selected_machine} RUL is low!\n"
            f"Current RUL: {latest_rul_value} hours"
        )
        send_telegram_alert(chat_id, telegram_message)

    if st.button("▶️ Start RUL Stream"):
        for _ in range(100):
            time.sleep(10)
            latest_rul = get_rul_data(selected_machine).tail(1)
            latest_rul_value = latest_rul["rul_pred"].iloc[-1] if not latest_rul.empty else float('inf')
            rul_chart.add_rows(latest_rul[["rul_pred"]])

            if latest_rul_value < alert_threshold:
                st.warning(f"⚠️ Warning! RUL is below {alert_threshold} hours! Maintenance is recommended.", icon="⚠️")
                send_email_alert(selected_machine, latest_rul_value)
                chat_id = -1002671447415
                telegram_message = (
                    f"⚠️ <b>Maintenance Alert</b>\n"
                    f"Machine {selected_machine} RUL is low!\n"
                    f"Current RUL: {latest_rul_value} hours"
                )
                send_telegram_alert(chat_id, telegram_message)

# --------------------------- 
# 📉 Tab 3 - Overview of All Machines
# --------------------------- 
with tab3:
    st.subheader("Latest RUL for All Machines (Sorted by Lowest RUL)")

    # Cache all RUL data
    @st.cache_data(ttl=60)
    def get_all_rul_data():
        all_rul_query = """
            SELECT * FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY machineid ORDER BY prediction_time DESC) as rn
                FROM predictions
            ) sub WHERE rn = 1
            ORDER BY rul_pred ASC
        """
        return pd.read_sql(all_rul_query, engine)

    latest_rul_all = get_all_rul_data()
    st.dataframe(latest_rul_all[["machineid", "prediction_time", "rul_pred"]])

    st.subheader("📈 Average RUL Trend (Optional)")
    avg_rul_query = """
        SELECT prediction_time, AVG(rul_pred) as avg_rul
        FROM predictions
        GROUP BY prediction_time
        ORDER BY prediction_time
    """
    avg_rul_df = pd.read_sql(avg_rul_query, engine)
    avg_rul_df["prediction_time"] = pd.to_datetime(avg_rul_df["prediction_time"])
    avg_rul_df.set_index("prediction_time", inplace=True)
    st.line_chart(avg_rul_df[["avg_rul"]])

# --------------------------- 
# 🤖 AI Chat Assistant
# --------------------------- 
from mistral_query import build_prompt, query_mistral

# Session init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create chat toggle box (HTML + JS safely embedded)
components.html("""
<style>
#chat-button {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 14px;
    border-radius: 50%;
    font-size: 20px;
    cursor: pointer;
    z-index: 9999;
}
#chat-window {
    display: none;
    position: fixed;
    bottom: 80px;
    right: 20px;
    width: 350px;
    max-height: 500px;
    background-color: white;
    border: 1px solid #ccc;
    border-radius: 12px;
    padding: 10px;
    overflow-y: auto;
    z-index: 9998;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
</style>

<button id="chat-button" onclick="toggleChat()">💬</button>
<div id="chat-window">
    <h4>🤖 AI Maintenance Assistant</h4>
    <div id="chat-content"></div>
</div>

<script>
function toggleChat() {
    var win = document.getElementById("chat-window");
    if (win.style.display === "none") {
        win.style.display = "block";
    } else {
        win.style.display = "none";
    }
}
</script>
""", height=0)  # Zero height since it’s floating

# Display chat input and history
st.write("## AI Assistant Chatbox")
for sender, msg in st.session_state.chat_history:
    align = "user" if sender == "user" else "assistant"
    with st.chat_message(align):
        st.markdown(msg)

# Input
user_input = st.chat_input("Type your question here")

def extract_machine_id(text):
    match = re.search(r'\b(?:machine\s*)?(\d{1,6})\b', text, re.IGNORECASE)
    return match.group(1) if match else None

# Process user input
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.last_user_input = user_input  # Save to detect processing needed
    st.rerun()

if "last_user_input" in st.session_state:
    user_input = st.session_state.last_user_input
    del st.session_state.last_user_input

    with st.spinner("🤖 Thinking..."):
        machine_id = extract_machine_id(user_input)
        try:
            if machine_id:
                prompt = build_prompt(machine_id, user_input, engine)
            else:
                prompt = build_prompt(None, user_input, engine)
            answer = query_mistral(prompt)
        except Exception as e:
            answer = f"⚠️ Error: {e}"

    st.session_state.chat_history.append(("assistant", answer))
    st.rerun()
