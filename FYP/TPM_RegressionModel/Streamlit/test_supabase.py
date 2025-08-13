from sqlalchemy import create_engine, text

# --- Replace with your actual connection string from Supabase ---
DB_URL = "postgresql://postgres:BigRoot%40123%40@db.gwvaqtikukibtwmtbpar.supabase.co:5432/postgres?sslmode=require"

print("🔄 Attempting to connect to Supabase...")

try:
    # Create engine
    engine = create_engine(DB_URL)
    
    # Test connection
    with engine.connect() as conn:
        print("✅ Connected successfully!")
        
        # Run a simple query
        result = conn.execute(text("SELECT NOW();"))
        print("🕒 Server time:", list(result))
        
except Exception as e:
    print("❌ Connection failed!")
    print("Error:", e)
