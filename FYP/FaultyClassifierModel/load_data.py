# load_data.py
import psycopg2
import pandas as pd

def load_machine_data():
    conn = psycopg2.connect(
        dbname="TPMDB",        # Replace with your actual database name
        user="postgres",       # Replace with your actual user
        password="root",       # Replace with your actual password
        host="localhost",      # Replace with your actual host (localhost or IP address)
        port="5433"            # Replace with your actual port
    )
    
    query = "SELECT * FROM machine_data"
    df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    # Print the dataframe to verify the data
    print(df)
    
    return df
