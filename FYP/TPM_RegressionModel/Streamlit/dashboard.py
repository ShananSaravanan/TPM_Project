import streamlit as st
import pandas as pd
import time
from sqlalchemy import create_engine, text
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import streamlit.components.v1 as components
import logging
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------- 
# 🎛 Page & DB Setup
# --------------------------- 
st.set_page_config(page_title="Professional Preventive Maintenance Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for light mode and professional styling
st.markdown("""
    <style>
    /* Force light mode */
    .stApp {
        background-color: #ffffff;
        color: #333333;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px;
    }
    .sidebar-button {
        display: block;
        width: 100%;
        padding: 10px;
        margin: 5px 0;
        background-color: #e9ecef;
        border: none;
        border-radius: 5px;
        text-align: left;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .sidebar-button:hover {
        background-color: #dee2e6;
    }
    .sidebar-button.active {
        background-color: #007bff;
        color: white;
    }
    /* Header styling */
    .header {
        background-color: #007bff;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    /* Card styling for graphs */
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    /* Chat assistant styling */
    #chat-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #007bff;
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
        background-color: #ffffff;
        border: 1px solid #ccc;
        border-radius: 12px;
        padding: 10px;
        overflow-y: auto;
        z-index: 9998;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Database connection
try:
    engine = create_engine("postgresql://postgres:root@localhost:5433/AzureTPMDB")
    logger.info("Database engine initialized successfully")
except Exception as e:
    st.error(f"Failed to connect to database: {str(e)}")
    logger.error(f"Database connection error: {e}")
    raise e

# --------------------------- 
# 🔐 Authentication Setup
# --------------------------- 
def create_users_table():
    """Create users table if it doesn't exist or verify schema."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'users' AND column_name = 'password'
            """)).fetchone()
            if result and result[1].lower() != 'character varying':
                logger.warning("Incorrect password column type detected. Dropping and recreating users table.")
                conn.execute(text("DROP TABLE IF EXISTS users"))
                conn.commit()
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(100) NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
            logger.info("Users table created or verified")
    except Exception as e:
        logger.error(f"Failed to create users table: {e}")
        st.error(f"Database error: {e}")

def create_maintenance_history_table():
    """Create maintenance history table for analytics."""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS maintenance_history (
                    id SERIAL PRIMARY KEY,
                    machineid VARCHAR(50) NOT NULL,
                    maintenance_date DATE NOT NULL,
                    maintenance_type VARCHAR(50) NOT NULL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
            logger.info("Maintenance history table created or verified")
    except Exception as e:
        logger.error(f"Failed to create maintenance history table: {e}")

def validate_email(email):
    """Validate email format."""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def register_user(username, password, email):
    """Register a new user in the database with plain text password."""
    try:
        if not username or not password or not email:
            logger.warning("Registration failed: Empty fields provided")
            return False
        if not validate_email(email):
            logger.warning(f"Registration failed: Invalid email format {email}")
            return False
        username_lower = username.lower()
        email_lower = email.lower()
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO users (username, password, email) 
                VALUES (:username, :password, :email)
            """), {"username": username_lower, "password": password, "email": email_lower})
            conn.commit()
        logger.info(f"User {username_lower} registered successfully")
        return True
    except Exception as e:
        logger.error(f"Registration failed for {username_lower}: {e}")
        return False

def login_user(username, password):
    """Authenticate a user with plain text password."""
    try:
        if not username or not password:
            logger.warning("Login failed: Empty username or password")
            return False
        username_lower = username.lower()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT password FROM users WHERE username = :username
            """), {"username": username_lower}).fetchone()
            logger.debug(f"Login query for {username_lower} returned: {result}")
            if result:
                stored_password = result[0]
                if password == stored_password:
                    logger.info(f"User {username_lower} logged in successfully")
                    return True
                else:
                    logger.warning(f"Invalid password for {username_lower}. Entered: {password}, Stored: {stored_password}")
                    return False
            else:
                logger.warning(f"User {username_lower} not found")
                return False
    except Exception as e:
        logger.error(f"Login error for {username_lower}: {e}")
        return False

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_menu' not in st.session_state:
    st.session_state.selected_menu = "Dashboard"

# Create tables
create_users_table()
create_maintenance_history_table()

# --------------------------- 
# 🔐 Login/Register Page
# --------------------------- 
if not st.session_state.logged_in:
    st.title("🔐 Professional Preventive Maintenance Dashboard")
    auth_option = st.selectbox("Choose Action", ["Login", "Register"])

    if auth_option == "Login":
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            if login_button:
                if not username or not password:
                    st.error("Please fill in both username and password.")
                    logger.warning("Login attempt failed: Empty fields")
                elif login_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username.lower()
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please check your credentials.")
                    logger.warning(f"Login attempt failed for {username.lower()}")

    else:  # Register
        with st.form("register_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            register_button = st.form_submit_button("Register")
            if register_button:
                if not username or not password or not email:
                    st.error("Please fill in all fields.")
                    logger.warning("Registration failed: Empty fields")
                elif not validate_email(email):
                    st.error("Please enter a valid email address.")
                    logger.warning(f"Registration failed: Invalid email {email}")
                elif register_user(username, password, email):
                    st.success("Registration successful! Please login.")
                    logger.info(f"User {username.lower()} registered successfully")
                else:
                    st.error("Registration failed. Username or email may already exist.")
                    logger.warning(f"Registration failed for {username.lower()}")

    # Debug tool
    with st.expander("Debug Authentication"):
        debug_username = st.text_input("Debug Username")
        debug_password = st.text_input("Debug Password", type="password")
        if st.button("Run Debug"):
            try:
                with engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT username, password, email, created_at 
                        FROM users 
                        WHERE username = :username
                    """), {"username": debug_username.lower()}).fetchone()
                    if result:
                        stored_password = result[1]
                        password_match = debug_password == stored_password
                        debug_result = {
                            "user_found": True,
                            "password_match": password_match,
                            "stored_password": stored_password,
                            "entered_password": debug_password,
                            "username": result[0],
                            "email": result[2],
                            "created_at": str(result[3])
                        }
                        logger.info(f"Debug for {debug_username.lower()}: {debug_result}")
                    else:
                        debug_result = {"user_found": False, "password_match": False}
                        logger.info(f"Debug for {debug_username.lower()}: User not found")
                    st.write(debug_result)
            except Exception as e:
                debug_result = {"error": str(e)}
                logger.error(f"Debug error: {e}")
                st.write(debug_result)

else:
    # --------------------------- 
    # 📋 Main Dashboard with Sidebar
    # --------------------------- 
    st.markdown('<div class="header"><h2>Preventive Maintenance Dashboard</h2></div>', unsafe_allow_html=True)
    st.sidebar.markdown(f"<h3>Welcome, {st.session_state.username}</h3>", unsafe_allow_html=True)

    # Sidebar menu with buttons
    menu_items = [
        ("Dashboard", "📊"),
        ("Live Telemetry", "🔋"),
        ("RUL Watch", "⏳"),
        ("Maintenance Scheduler", "🗓"),
        ("Assets", "🛠"),
        ("Work Orders", "📋"),
        ("Management", "👷")
    ]
    for item, icon in menu_items:
        if st.sidebar.button(f"{item} {icon}", key=item, help=f"Navigate to {item}"):
            st.session_state.selected_menu = item
        if st.session_state.selected_menu == item:
            st.sidebar.markdown(f'<style>.sidebar-button[key="{item}"] {{ background-color: #007bff; color: white; }}</style>', unsafe_allow_html=True)
    st.sidebar.button("Logout", on_click=lambda: [setattr(st.session_state, 'logged_in', False), setattr(st.session_state, 'username', None)])

    # --------------------------- 
    # 🔧 Common Functions
    # --------------------------- 
    def send_telegram_alert(chat_id, message_text):
        bot_token = "7688190828:AAHF3QU70K5A8djY_d3RViMH-uA6Nsl6LL0"
        send_message_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message_text, "parse_mode": "HTML"}
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
        sender_password = "aswv tqus gstv wnfc"
        recipient_email = "shanansaravanan03@gmail.com"
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

    # --------------------------- 
    # 📊 Dashboard
    # --------------------------- 
    if st.session_state.selected_menu == "Dashboard":
        st.title("📊 Analytics Dashboard")
        st.markdown("Comprehensive insights into machine health, performance, and maintenance needs.")

        # Filters
        col1, col2 = st.columns([3, 1])
        with col1:
            machine_ids = pd.read_sql("SELECT DISTINCT machineid FROM predictions", engine)["machineid"].tolist()
            selected_machines = st.multiselect("Filter Machines", machine_ids, default=machine_ids[:5])
        with col2:
            date_range = st.date_input("Select Date Range", [datetime.now().date(), datetime.now().date()])

        # Summary Metrics
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            critical_machines = len(pd.read_sql("SELECT DISTINCT machineid FROM predictions WHERE rul_pred < 200", engine))
            st.metric("Critical Machines", critical_machines, delta=-1)
        with col2:
            avg_rul = pd.read_sql("SELECT AVG(rul_pred) as avg_rul FROM predictions", engine)["avg_rul"].iloc[0]
            st.metric("Average RUL (Hours)", f"{avg_rul:.1f}", delta=10)
        with col3:
            total_machines = len(machine_ids)
            st.metric("Total Machines", total_machines)
        st.markdown('</div>', unsafe_allow_html=True)

        # Machine Health Score Gauge
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Machine Health Scores")
        health_scores = pd.read_sql("SELECT machineid, rul_pred FROM predictions WHERE prediction_time = (SELECT MAX(prediction_time) FROM predictions)", engine)
        health_scores["health_score"] = 100 * (health_scores["rul_pred"] / health_scores["rul_pred"].max())
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_machine = st.selectbox("Select Machine for Health Score", health_scores["machineid"])
            score = health_scores[health_scores["machineid"] == selected_machine]["health_score"].iloc[0]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': f"Health Score: Machine {selected_machine}"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#007bff"},
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 30}}))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(health_scores, x="machineid", y="health_score", title="Health Scores Across Machines")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Failure Risk Trend
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Failure Risk Trend")
        rul_data = pd.read_sql("SELECT prediction_time, AVG(rul_pred) as avg_rul FROM predictions GROUP BY prediction_time ORDER BY prediction_time", engine)
        rul_data["failure_risk"] = 100 * (1 - rul_data["avg_rul"] / rul_data["avg_rul"].max())
        fig = px.line(rul_data, x="prediction_time", y="failure_risk", title="Average Failure Risk Over Time")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Telemetry Correlation Heatmap
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Telemetry Correlation Analysis")
        telemetry = pd.read_sql("SELECT volt, rotate, pressure, vibration FROM telemetry WHERE machineid IN :machines", engine, params={"machines": tuple(selected_machines)})
        corr_matrix = telemetry.corr()
        fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap of Telemetry Metrics")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Maintenance History
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Maintenance History")
        maintenance_data = pd.read_sql("SELECT machineid, maintenance_date, maintenance_type FROM maintenance_history WHERE machineid IN :machines", engine, params={"machines": tuple(selected_machines)})
        if not maintenance_data.empty:
            fig = px.histogram(maintenance_data, x="machineid", color="maintenance_type", title="Maintenance Actions by Machine")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No maintenance history available. Schedule maintenance to populate this chart.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------- 
    # 🔋 Live Telemetry
    # --------------------------- 
    elif st.session_state.selected_menu == "Live Telemetry":
        st.title("🔋 Live Telemetry Viewer")
        machine_ids = pd.read_sql("SELECT DISTINCT machineid FROM predictions", engine)["machineid"].tolist()
        selected_machine = st.selectbox("🔧 Select Machine", machine_ids)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"Telemetry Trend for Machine {selected_machine}")
        telemetry_option = st.selectbox("Select telemetry type", ["volt", "rotate", "pressure", "vibration"])
        tele_query = text("SELECT * FROM telemetry WHERE machineid = :machine_id ORDER BY datetime")
        telemetry = pd.read_sql(tele_query, engine, params={"machine_id": selected_machine})
        telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])
        telemetry.set_index("datetime", inplace=True)
        fig = px.line(telemetry.tail(20), y=telemetry_option, title=f"{telemetry_option.capitalize()} Trend")
        st.plotly_chart(fig, use_container_width=True)
        if st.button("▶️ Start Telemetry Stream"):
            placeholder = st.empty()
            for _ in range(100):
                time.sleep(10)
                latest_query = text("""
                    SELECT * FROM telemetry 
                    WHERE machineid = :machine_id 
                    ORDER BY datetime DESC LIMIT 1
                """)
                latest = pd.read_sql(latest_query, engine, params={"machine_id": selected_machine})
                latest["datetime"] = pd.to_datetime(latest["datetime"])
                latest.set_index("datetime", inplace=True)
                telemetry = pd.concat([telemetry, latest])
                fig = px.line(telemetry.tail(20), y=telemetry_option, title=f"{telemetry_option.capitalize()} Trend")
                placeholder.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------- 
    # ⏳ RUL Watch
    # --------------------------- 
    elif st.session_state.selected_menu == "RUL Watch":
        st.title("⏳ RUL Watch")
        machine_ids = pd.read_sql("SELECT DISTINCT machineid FROM predictions", engine)["machineid"].tolist()
        selected_machine = st.selectbox("🔧 Select Machine", machine_ids)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"RUL Prediction Trend for Machine {selected_machine}")
        @st.cache_data(ttl=60)
        def get_rul_data(machine_id):
            rul_query = text("SELECT prediction_time, rul_pred FROM predictions WHERE machineid = :machine_id ORDER BY prediction_time")
            rul_data = pd.read_sql(rul_query, engine, params={"machine_id": machine_id})
            rul_data["prediction_time"] = pd.to_datetime(rul_data["prediction_time"])
            rul_data.set_index("prediction_time", inplace=True)
            return rul_data
        rul_data = get_rul_data(selected_machine)
        fig = px.line(rul_data.tail(30), y="rul_pred", title="RUL Prediction Trend")
        st.plotly_chart(fig, use_container_width=True)
        latest_rul_value = rul_data["rul_pred"].iloc[-1] if not rul_data.empty else float('inf')
        alert_threshold = 200
        if latest_rul_value < alert_threshold:
            st.warning(f"⚠️ Warning! RUL is below {alert_threshold} hours! Maintenance is recommended.", icon="⚠️")
            st.markdown("### 📨 Email Alert Sent for Low RUL!")
            send_email_alert(selected_machine, latest_rul_value)
            chat_id = -1002671447415
            telegram_message = (
                f"⚠️ <b>Maintenance Alert</b>\n"
                f"Machine {selected_machine} RUL is low!\n"
                f"Current RUL: {latest_rul_value} hours"
            )
            send_telegram_alert(chat_id, telegram_message)
        if st.button("▶️ Start RUL Stream"):
            placeholder = st.empty()
            for _ in range(100):
                time.sleep(10)
                latest_rul = get_rul_data(selected_machine).tail(1)
                latest_rul_value = latest_rul["rul_pred"].iloc[-1] if not latest_rul.empty else float('inf')
                rul_data = pd.concat([rul_data, latest_rul])
                fig = px.line(rul_data.tail(30), y="rul_pred", title="RUL Prediction Trend")
                placeholder.plotly_chart(fig, use_container_width=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------- 
    # 🗓 Maintenance Scheduler
    # --------------------------- 
    elif st.session_state.selected_menu == "Maintenance Scheduler":
        st.title("🗓 Maintenance Scheduler")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Schedule maintenance tasks for machines.")
        machine_ids = pd.read_sql("SELECT DISTINCT machineid FROM predictions", engine)["machineid"].tolist()
        selected_machine = st.selectbox("Select Machine", machine_ids)
        maintenance_date = st.date_input("Maintenance Date", datetime.now())
        maintenance_type = st.selectbox("Maintenance Type", ["Preventive", "Corrective", "Inspection"])
        notes = st.text_area("Notes")
        if st.button("Schedule Maintenance"):
            try:
                with engine.connect() as conn:
                    conn.execute(text("""
                        INSERT INTO maintenance_history (machineid, maintenance_date, maintenance_type, notes)
                        VALUES (:machineid, :maintenance_date, :maintenance_type, :notes)
                    """), {
                        "machineid": selected_machine,
                        "maintenance_date": maintenance_date,
                        "maintenance_type": maintenance_type,
                        "notes": notes
                    })
                    conn.commit()
                st.success(f"Maintenance scheduled for Machine {selected_machine} on {maintenance_date}!")
                logger.info(f"Maintenance scheduled for Machine {selected_machine}")
            except Exception as e:
                st.error(f"Failed to schedule maintenance: {e}")
                logger.error(f"Maintenance scheduling error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------- 
    # 🛠 Assets
    # --------------------------- 
    elif st.session_state.selected_menu == "Assets":
        st.title("🛠 Assets Management")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Manage machine assets.")
        st.subheader("Asset List")
        machine_ids = pd.read_sql("SELECT DISTINCT machineid FROM predictions", engine)["machineid"].tolist()
        st.dataframe(pd.DataFrame({"Machine ID": machine_ids}))
        st.subheader("Add New Asset")
        new_asset_id = st.text_input("New Machine ID")
        if st.button("Add Asset"):
            st.success(f"Asset {new_asset_id} added!")
            logger.info(f"Asset {new_asset_id} added (placeholder)")
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------- 
    # 📋 Work Orders
    # --------------------------- 
    elif st.session_state.selected_menu == "Work Orders":
        st.title("📋 Work Orders Management")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Manage work orders for maintenance tasks.")
        st.subheader("Create Work Order")
        machine_ids = pd.read_sql("SELECT DISTINCT machineid FROM predictions", engine)["machineid"].tolist()
        selected_machine = st.selectbox("Select Machine", machine_ids)
        task_description = st.text_area("Task Description")
        priority = st.selectbox("Priority", ["Low", "Medium", "High"])
        if st.button("Create Work Order"):
            st.success(f"Work order created for Machine {selected_machine}!")
            logger.info(f"Work order created for Machine {selected_machine} (placeholder)")
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------- 
    # 👷 Management
    # --------------------------- 
    elif st.session_state.selected_menu == "Management":
        st.title("👷 Management")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Manage teams, reports, and settings.")
        st.subheader("Team Management")
        st.write("Placeholder for team management functionality.")
        st.subheader("Reports")
        st.write("Placeholder for generating reports.")
        st.subheader("Settings")
        st.write("Placeholder for system settings.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------- 
    # 🤖 AI Chat Assistant
    # --------------------------- 
    from mistral_query import build_prompt, query_mistral
    components.html("""
    <button id="chat-button" onclick="toggleChat()">💬</button>
    <div id="chat-window">
        <h4>🤖 AI Maintenance Assistant</h4>
        <div id="chat-content"></div>
    </div>
    <script>
    function toggleChat() {
        var win = document.getElementById('chat-window');
        if (win.style.display === 'none') {
            win.style.display = 'block';
        } else {
            win.style.display = 'none';
        }
    }
    </script>
    """, height=0)
    st.write("## AI Assistant Chatbox")
    for sender, msg in st.session_state.chat_history:
        align = "user" if sender == "user" else "assistant"
        with st.chat_message(align):
            st.markdown(msg)
    user_input = st.chat_input("Type your question here")
    def extract_machine_id(text):
        match = re.search(r'\b(?:machine\s*)?(\d{1,6})\b', text, re.IGNORECASE)
        return match.group(1) if match else None
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.last_user_input = user_input
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