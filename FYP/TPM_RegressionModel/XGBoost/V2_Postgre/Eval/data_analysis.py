# data_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file path
file_path = 'FYP/TPM_RegressionModel/AzureDataset/training_dataset_with_rul.csv'

# Verify file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load data (use read_csv for CSV files; for Excel, uncomment the alternative line)
data = pd.read_csv(file_path)
# data = pd.read_excel(file_path, engine='openpyxl')  # Use this if the file is .xlsx

# Define RUL intervals (in hours)
bins = [0, 200, 400, 600, 800, 1000, 2000, float('inf')]
labels = ['0-200', '200-400', '400-600', '600-800', '800-1000', '1000-2000', '2000+']

# Count RUL occurrences in each interval
data['RUL_range'] = pd.cut(data['RUL_hours'], bins=bins, labels=labels, right=False)
rul_counts = data['RUL_range'].value_counts().sort_index()

# Print RUL range counts
print("RUL Distribution (in hours):")
for label, count in rul_counts.items():
    print(f"{label}: {count} machines")

# Create histogram
plt.figure(figsize=(10, 6))
sns.histplot(data['RUL_hours'], bins=bins[:-1], kde=False, color='skyblue')
plt.title('Histogram of RUL (Remaining Useful Life) in Hours')
plt.xlabel('RUL (hours)')
plt.ylabel('Number of Machines')
plt.xticks(bins[:-1])
plt.grid(True, axis='y')
plt.savefig('rul_histogram.png')
plt.close()

print("\nHistogram saved as 'rul_histogram.png'")