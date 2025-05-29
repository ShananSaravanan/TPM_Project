import pandas as pd
from sklearn.model_selection import train_test_split

# Load your full dataset
df = pd.read_csv("TPM_RegressionModel/AzureDataset/training_dataset_with_rul.csv")

# Optional: shuffle before splitting to ensure randomness
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train and test (e.g., 80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the test set
test_df.to_csv("TPM_RegressionModel/AzureDataset/test_set.csv", index=False)

print("✅ Test set created and saved!")
