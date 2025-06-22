import pandas as pd
import random
from datetime import datetime

# Path to your CSV file (update this to your actual file path)
CSV_FILE_PATH = "FYP/TPM_RegressionModel/AzureDataset/PdM_telemetry.csv"  # Replace with the actual path to your CSV file

# Analyze CSV dataset to find min and max values for parameters across all machines
def analyze_telemetry_csv(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        required_columns = ["volt", "rotate", "pressure", "vibration"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")
        
        # Calculate min and max for each parameter across all data
        ranges = {
            "volt": (df["volt"].min(), df["volt"].max()),
            "rotate": (df["rotate"].min(), df["rotate"].max()),
            "pressure": (df["pressure"].min(), df["pressure"].max()),
            "vibration": (df["vibration"].min(), df["vibration"].max())
        }
        return ranges
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error analyzing CSV: {str(e)}")
        return None

# Generate mock telemetry data based on analyzed ranges without rounding
def generate_mock_telemetry(ranges):
    return {
        "datetime": datetime.now().isoformat(),
        "volt": random.uniform(ranges["volt"][0], ranges["volt"][1]),
        "rotate": random.uniform(ranges["rotate"][0], ranges["rotate"][1]),
        "pressure": random.uniform(ranges["pressure"][0], ranges["pressure"][1]),
        "vibration": random.uniform(ranges["vibration"][0], ranges["vibration"][1])
    }

# Main execution
if __name__ == "__main__":
    # Analyze the CSV file
    telemetry_ranges = analyze_telemetry_csv(CSV_FILE_PATH)
    
    if telemetry_ranges:
        # Print the analyzed ranges
        print("Analyzed Ranges:")
        for param, (min_val, max_val) in telemetry_ranges.items():
            print(f"{param}: [{min_val}, {max_val}]")
        
        # Generate one mock telemetry entry representing all machines
        mock_data = generate_mock_telemetry(telemetry_ranges)
        print("\nMock Telemetry Data (representing all machines):", mock_data)
    else:
        print("Failed to generate mock data due to errors in CSV analysis.")