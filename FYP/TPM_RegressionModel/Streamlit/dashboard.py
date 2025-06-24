import streamlit as st
import pandas as pd
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
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------- 
# 🎛 Page & DB Setup
# --------------------------- 
st.set_page_config(page_title="AIPM+ Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for light mode, professional styling, and menu
st.markdown("""
    <style>
    /* Force light mode */
    .stApp {
        background-color: #ffffff;
        color: #333333;
    }
    /* Sidebar styling */
    .css-1lcbmhc {
        background-color: #2c2f33;
        padding: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #2c2f33;
        padding: 0;
    }
    /* Menu button styling */
    button[data-testid="stButton"] {
        display: block;
        width: 100%;
        padding: 10px 15px;
        color: #ffffff;
        text-align: left;
        border: none;
        background: none;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    button[data-testid="stButton"]:hover {
        background-color: #7289da;
        color: #ffffff;
    }
    /* Active menu button styling */
    button[data-testid="stButton"].active {
        background-color: #7289da;
        color: #ffffff;
    }
    /* Logout button styling */
    button[data-testid="stButton"][key="Logout"] {
        background-color: #ff0000;
        color: #ffffff;
    }
    button[data-testid="stButton"][key="Logout"]:hover {
        background-color: #cc0000;
        color: #ffffff;
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
    /* Card styling for graphs */
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
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
    """Create users table if it doesn't exist."""
    try:
        with engine.connect() as conn:
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

def create_work_orders_table():
    """Create work orders table if it doesn't exist."""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS work_orders (
                    id SERIAL PRIMARY KEY,
                    machineid INTEGER NOT NULL,
                    task_description TEXT NOT NULL,
                    priority VARCHAR(20) NOT NULL,
                    status VARCHAR(20) DEFAULT 'Pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
            logger.info("Work orders table created or verified")
    except Exception as e:
        logger.error(f"Failed to create work orders table: {e}")

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
    st.session_state.selected_menu = None  # Set to None initially, will be set post-login
if 'alert_email' not in st.session_state:
    st.session_state.alert_email = "shanansaravanan03@gmail.com"  # Default email
if 'rerun_trigger' not in st.session_state:
    st.session_state.rerun_trigger = False

# Create tables (only new ones, no dropping existing tables)
create_users_table()
create_work_orders_table()

# --------------------------- 
# 🔐 Login/Register Page
# --------------------------- 
if not st.session_state.logged_in:
    st.title("🔐 AIPM+ Dashboard")
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
                    st.session_state.selected_menu = "Dashboard 📊"  # Set default menu after login
                    st.session_state.rerun_trigger = True
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

else:
    # --------------------------- 
    # 📋 Main Dashboard with Sidebar
    # --------------------------- 
    st.sidebar.markdown("<h3 style='color: black;'>AIPM+ System</h3>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<h3 style='color: black;'>Welcome, {st.session_state.username}</h3>", unsafe_allow_html=True)

    # Sidebar menu with buttons
    menu_items = [
        "Dashboard 📊",
        "Live Telemetry 🔋",
        "RUL Watch ⏳",
        "Maintenance Scheduler 🗓",
        "Assets 🛠",
        "Work Orders 📋",
        "Order Requests 📬",
        "Reports 📈",
        "Account 👤",
        "Configuration ⚙️"
    ]

    # Set default menu if not set post-login
    if st.session_state.selected_menu is None:
        st.session_state.selected_menu = "Dashboard 📊"

    for item in menu_items:
        if st.sidebar.button(item, key=item, help=f"Navigate to {item.split(' ')[0]}"):
            st.session_state.selected_menu = item
            st.session_state.rerun_trigger = True
        if st.session_state.selected_menu == item:
            st.sidebar.markdown(f'<style>button[data-testid="stButton"][key="{item}"] {{ background-color: #7289da; color: #ffffff; }}</style>', unsafe_allow_html=True)

    def logout():
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.selected_menu = None
        st.session_state.rerun_trigger = True

    st.sidebar.button("Logout", on_click=logout, key="Logout")

    # Trigger rerun based on session state flag
    if st.session_state.rerun_trigger:
        st.session_state.rerun_trigger = False
        st.rerun()

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

    def send_email_alert(machine_id, rul_value, recipient_email):
        sender_email = "shananmessi10@gmail.com"
        sender_password = "aswv tqus gstv wnfc"
        subject = f"⚠️ Maintenance Alert: Machine {machine_id} RUL Below Threshold"
        body = f"Warning! The Remaining Useful Life (RUL) of Machine {machine_id} has dropped below the threshold.\n\nCurrent RUL: {rul_value} hours\nThreshold: {alert_threshold} hours"
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
    if st.session_state.selected_menu == "Dashboard 📊":
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
        col1, col2, col3 = st.columns(3)
        with col1:
            critical_machines = len(pd.read_sql("SELECT DISTINCT machineid FROM predictions WHERE rul_pred < 200", engine))
            st.metric("Critical Machines", critical_machines, delta=-1)
        with col2:
            avg_rul = pd.read_sql("SELECT AVG(rul_pred) as avg_rul FROM predictions", engine)["avg_rul"].iloc[0]
            st.metric("Average RUL (Hours)", f"{avg_rul:.1f}", delta=10)
        with col3:
            total_machines = len(pd.read_sql("SELECT DISTINCT machineid FROM machines", engine))
            st.metric("Total Machines", total_machines)

        # Machine Health Score Gauge
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
                       'steps': [
                           {'range': [0, 30], 'color': "red"},
                           {'range': [30, 70], 'color': "orange"},
                           {'range': [70, 100], 'color': "green"}
                       ]}))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(health_scores, x="machineid", y="health_score", title="Health Scores Across Machines")
            st.plotly_chart(fig, use_container_width=True)

        # Failure Risk Trend
        st.subheader("Failure Risk Trend")
        rul_data = pd.read_sql("SELECT prediction_time, AVG(rul_pred) as avg_rul FROM predictions GROUP BY prediction_time ORDER BY prediction_time", engine)
        rul_data["failure_risk"] = 100 * (1 - rul_data["avg_rul"] / rul_data["avg_rul"].max())
        fig = px.line(rul_data, x="prediction_time", y="failure_risk", title="Average Failure Risk Over Time")
        st.plotly_chart(fig, use_container_width=True)

        # Telemetry Correlation Heatmap
        st.subheader("Telemetry Correlation Analysis")
        if selected_machines:
            placeholders = ", ".join(["%s" for _ in selected_machines])
            query = f"SELECT volt, rotate, pressure, vibration FROM telemetry WHERE machineid IN ({placeholders})"
            telemetry = pd.read_sql(query, engine, params=tuple(selected_machines))
            corr_matrix = telemetry.corr()
            fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap of Telemetry Metrics")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No machines selected for correlation analysis.")

        # Maintenance History
        st.subheader("Maintenance History")
        if selected_machines:
            placeholders = ", ".join(["%s" for _ in selected_machines])
            query = f"SELECT machineid, datetime, comp FROM maintenance WHERE machineid IN ({placeholders})"
            maintenance_data = pd.read_sql(query, engine, params=tuple(selected_machines))
            if not maintenance_data.empty:
                fig = px.histogram(maintenance_data, x="machineid", color="comp", title="Maintenance Actions by Machine")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No maintenance history available. Schedule maintenance to populate this chart.")
        else:
            st.info("No machines selected for maintenance history.")

    # --------------------------- 
    # 🔋 Live Telemetry
    # --------------------------- 
    elif st.session_state.selected_menu == "Live Telemetry 🔋":
        st.title("🔋 Live Telemetry Viewer")
        machine_ids = pd.read_sql("SELECT DISTINCT machineid FROM predictions", engine)["machineid"].tolist()
        selected_machine = st.selectbox("🔧 Select Machine", machine_ids)
        st.subheader(f"Telemetry Trend for Machine {selected_machine}")
        telemetry_option = st.selectbox("Select telemetry type", ["volt", "rotate", "pressure", "vibration"])
        tele_query = text("SELECT * FROM telemetry WHERE machineid = :machine_id ORDER BY datetime")
        telemetry = pd.read_sql(tele_query, engine, params={"machine_id": selected_machine})
        telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])
        telemetry.set_index("datetime", inplace=True)
        fig = px.line(telemetry.tail(20), y=telemetry_option, title=f"{telemetry_option.capitalize()} Trend")
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------- 
    # ⏳ RUL Watch
    # --------------------------- 
    elif st.session_state.selected_menu == "RUL Watch ⏳":
        st.title("⏳ RUL Watch")
        machine_ids = pd.read_sql("SELECT DISTINCT machineid FROM predictions", engine)["machineid"].tolist()
        selected_machine = st.selectbox("🔧 Select Machine", machine_ids)
        
        st.subheader(f"RUL Prediction Trend for Machine {selected_machine}")
        @st.cache_data(ttl=60)
        def get_rul_data(machine_id):
            rul_query = text("SELECT prediction_time, rul_pred FROM predictions WHERE machineid = :machine_id ORDER BY prediction_time DESC LIMIT 2")
            rul_data = pd.read_sql(rul_query, engine, params={"machine_id": machine_id})
            rul_data["prediction_time"] = pd.to_datetime(rul_data["prediction_time"])
            return rul_data

        rul_data = get_rul_data(selected_machine)
        if not rul_data.empty:
            recent_rul = rul_data.iloc[0]["rul_pred"]
            col1, col2 = st.columns(2)
            with col1:
                if len(rul_data) > 1:
                    last_rul = rul_data.iloc[1]["rul_pred"]
                    rul_change = recent_rul - last_rul
                    st.metric("RUL Predicted", f"{recent_rul:.1f}", delta=f"{rul_change:+.1f}")
                else:
                    st.metric("RUL Predicted", f"{recent_rul:.1f}", delta="-")
            with col2:
                if len(rul_data) > 1:
                    st.metric("Last RUL", f"{last_rul}")
                else:
                    st.metric("Last RUL", "-")

        fig = px.line(rul_data, x="prediction_time", y="rul_pred", title="RUL Prediction Trend")
        st.plotly_chart(fig, use_container_width=True)

        # Average RUL Over Time for All Machines
        st.subheader("Average RUL Over Time Across All Machines")
        @st.cache_data(ttl=60)
        def get_average_rul_over_time():
            query = "SELECT prediction_time, AVG(rul_pred) as avg_rul FROM predictions GROUP BY prediction_time ORDER BY prediction_time"
            avg_rul_data = pd.read_sql(query, engine)
            avg_rul_data["prediction_time"] = pd.to_datetime(avg_rul_data["prediction_time"])
            return avg_rul_data
        avg_rul_over_time = get_average_rul_over_time()
        if not avg_rul_over_time.empty:
            fig = px.line(avg_rul_over_time, x="prediction_time", y="avg_rul", title="Average RUL Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No RUL data available for average over time.")

        # RUL Distribution Bar Chart
        st.subheader("RUL Distribution Across Machines")
        @st.cache_data(ttl=60)
        def get_rul_distribution():
            query = "SELECT machineid, rul_pred FROM predictions WHERE prediction_time = (SELECT MAX(prediction_time) FROM predictions)"
            rul_dist = pd.read_sql(query, engine)
            return rul_dist
        rul_dist_data = get_rul_distribution()
        if not rul_dist_data.empty:
            fig = px.bar(rul_dist_data, x="machineid", y="rul_pred", title="RUL Distribution Across Machines")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No RUL data available for distribution.")

    # --------------------------- 
    # 🗓 Maintenance Scheduler
    # --------------------------- 
    elif st.session_state.selected_menu == "Maintenance Scheduler 🗓":
        st.title("🗓 Maintenance Scheduler")
        
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
                        INSERT INTO maintenance (datetime, machineid, comp)
                        VALUES (:datetime, :machineid, :comp)
                    """), {
                        "datetime": maintenance_date,
                        "machineid": selected_machine,
                        "comp": maintenance_type
                    })
                    conn.commit()
                st.success(f"Maintenance scheduled for Machine {selected_machine} on {maintenance_date}!")
                logger.info(f"Maintenance scheduled for Machine {selected_machine}")
            except Exception as e:
                st.error(f"Failed to schedule maintenance: {e}")
                logger.error(f"Maintenance scheduling error: {e}")

    # --------------------------- 
    # 🛠 Assets
    # --------------------------- 
    elif st.session_state.selected_menu == "Assets 🛠":
        st.title("🛠 Assets Management")
        
        st.write("View and manage machine assets details.")
        st.subheader("Machine Details")
        try:
            machines = pd.read_sql("SELECT machineid, model, age FROM machines", engine)
            if not machines.empty:
                st.dataframe(machines)
            else:
                st.info("No machine details available in the database.")
        except Exception as e:
            st.error(f"Failed to load machine details: {e}")
            logger.error(f"Machine details loading error: {e}")

        st.subheader("Add New Machine")
        with st.form("add_machine_form"):
            new_machine_id = st.number_input("Machine ID", min_value=0, step=1)
            model = st.text_input("Model")
            age = st.number_input("Age", min_value=0, step=1)
            if st.form_submit_button("Add Machine"):
                try:
                    with engine.connect() as conn:
                        conn.execute(text("""
                            INSERT INTO machines (machineid, model, age)
                            VALUES (:machineid, :model, :age)
                        """), {
                            "machineid": new_machine_id,
                            "model": model,
                            "age": age
                        })
                        conn.commit()
                    st.success(f"Machine {new_machine_id} added successfully!")
                    logger.info(f"Machine {new_machine_id} added")
                except Exception as e:
                    st.error(f"Failed to add machine: {e}")
                    logger.error(f"Machine addition error: {e}")

    # --------------------------- 
    # 📋 Work Orders
    # --------------------------- 
    elif st.session_state.selected_menu == "Work Orders 📋":
        st.title("📋 Work Orders Management")
        
        st.write("Manage work orders for maintenance tasks.")
        st.subheader("Create Work Order")
        machine_ids = pd.read_sql("SELECT DISTINCT machineid FROM predictions", engine)["machineid"].tolist()
        selected_machine = st.selectbox("Select Machine", machine_ids)
        task_description = st.text_area("Task Description")
        priority = st.selectbox("Priority", ["Low", "Medium", "High"])
        if st.button("Create Work Order"):
            try:
                with engine.connect() as conn:
                    conn.execute(text("""
                        INSERT INTO work_orders (machineid, task_description, priority)
                        VALUES (:machineid, :task_description, :priority)
                    """), {
                        "machineid": selected_machine,
                        "task_description": task_description,
                        "priority": priority
                    })
                    conn.commit()
                st.success(f"Work order created for Machine {selected_machine}!")
                logger.info(f"Work order created for Machine {selected_machine}")
            except Exception as e:
                st.error(f"Failed to create work order: {e}")
                logger.error(f"Work order creation error: {e}")

    # --------------------------- 
    # 📬 Order Requests
    # --------------------------- 
    elif st.session_state.selected_menu == "Order Requests 📬":
        st.title("📬 Order Requests")
        
        st.write("View all requested work orders.")
        st.subheader("Work Orders List")
        try:
            work_orders = pd.read_sql("SELECT id, machineid, task_description, priority, status, created_at FROM work_orders ORDER BY created_at DESC", engine)
            if not work_orders.empty:
                st.dataframe(work_orders)
            else:
                st.info("No work orders found.")
        except Exception as e:
            st.error(f"Failed to load work orders: {e}")
            logger.error(f"Work orders loading error: {e}")

    # --------------------------- 
    # 📈 Reports
    # --------------------------- 
    elif st.session_state.selected_menu == "Reports 📈":
        st.title("📈 Reports")
        
        st.write("Generate and view maintenance reports.")
        st.subheader("Maintenance Report")
        report_type = st.selectbox("Select Report Type", ["Maintenance History", "RUL Summary"])
        if report_type == "Maintenance History":
            try:
                maintenance_data = pd.read_sql("SELECT machineid, datetime, comp FROM maintenance ORDER BY datetime DESC", engine)
                if not maintenance_data.empty:
                    st.dataframe(maintenance_data)
                    fig = px.histogram(maintenance_data, x="machineid", color="comp", title="Maintenance Actions by Machine")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No maintenance history available.")
            except Exception as e:
                st.error(f"Failed to load maintenance history: {e}")
                logger.error(f"Maintenance history loading error: {e}")
        elif report_type == "RUL Summary":
            try:
                rul_data = pd.read_sql("SELECT machineid, rul_pred, prediction_time FROM predictions WHERE prediction_time = (SELECT MAX(prediction_time) FROM predictions)", engine)
                if not rul_data.empty:
                    st.dataframe(rul_data)
                    fig = px.bar(rul_data, x="machineid", y="rul_pred", title="Current RUL by Machine")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No RUL predictions available.")
            except Exception as e:
                st.error(f"Failed to load RUL data: {e}")
                logger.error(f"RUL data loading error: {e}")

    # --------------------------- 
    # 👤 Account
    # --------------------------- 
    elif st.session_state.selected_menu == "Account 👤":
        st.title("👤 Account Settings")
        
        st.write("Manage your account details.")
        st.subheader("User Information")
        try:
            with engine.connect() as conn:
                user_data = conn.execute(text("""
                    SELECT username, email, created_at 
                    FROM users 
                    WHERE username = :username
                """), {"username": st.session_state.username}).fetchone()
                if user_data:
                    st.write(f"Username: {user_data[0]}")
                    st.write(f"Email: {user_data[1]}")
                    st.write(f"Account Created: {user_data[2]}")
        except Exception as e:
            st.error(f"Failed to load user information: {e}")
            logger.error(f"User information loading error: {e}")

        st.subheader("Update Password")
        with st.form("update_password"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            if st.form_submit_button("Update Password"):
                if new_password != confirm_password:
                    st.error("New passwords do not match.")
                elif login_user(st.session_state.username, current_password):
                    try:
                        with engine.connect() as conn:
                            conn.execute(text("""
                                UPDATE users 
                                SET password = :new_password 
                                WHERE username = :username
                            """), {
                                "new_password": new_password,
                                "username": st.session_state.username
                            })
                            conn.commit()
                        st.success("Password updated successfully!")
                        logger.info(f"Password updated for user {st.session_state.username}")
                    except Exception as e:
                        st.error(f"Failed to update password: {e}")
                        logger.error(f"Password update error: {e}")
                else:
                    st.error("Current password is incorrect.")

    # --------------------------- 
    # ⚙️ Configuration
    # --------------------------- 
    elif st.session_state.selected_menu == "Configuration ⚙️":
        st.title("⚙️ Configuration")
        
        st.write("Configure system settings and parameters.")
        st.subheader("General Settings")
        st.write("Placeholder for general configuration options.")
        st.subheader("Alert Thresholds")
        alert_threshold = st.number_input("RUL Alert Threshold (hours)", min_value=0, value=200)
        if st.button("Save Threshold"):
            st.success(f"RUL Alert Threshold set to {alert_threshold} hours!")
            logger.info(f"RUL Alert Threshold updated to {alert_threshold}")
        
        st.subheader("Email Settings")
        new_alert_email = st.text_input("Alert Email Address", value=st.session_state.alert_email)
        if st.button("Update Alert Email"):
            if validate_email(new_alert_email):
                st.session_state.alert_email = new_alert_email
                st.success(f"Alert email updated to {new_alert_email}!")
                logger.info(f"Alert email updated to {new_alert_email}")
            else:
                st.error("Please enter a valid email address.")
                logger.warning(f"Invalid email attempt: {new_alert_email}")

    # --------------------------- 
    # 🤖 AI Chat Assistant (only for other pages)
    # --------------------------- 
    if st.session_state.selected_menu not in ["Account 👤", "Configuration ⚙️"]:
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
        st.write("## AI Assistant Chatbot")
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
            st.session_state.rerun_trigger = True
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
            st.session_state.rerun_trigger = True