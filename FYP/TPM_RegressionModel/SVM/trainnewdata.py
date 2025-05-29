import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the data
df = pd.read_csv('AzureDataset/telemetry_with_labels.csv')

# Define features and target
features = ['volt', 'rotate', 'pressure', 'vibration']
target = 'failure_label'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# Step 1: Use class weights for imbalanced data
class_weights = {0: 1, 1: 10}  # Adjust the weights for better performance

# Step 2: Define the Random Forest with class weights
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42, class_weight=class_weights)

# Step 3: Train the model with class weights and track progress using tqdm
for _ in tqdm(range(1), desc="Training Random Forest"):
    rf_classifier.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = rf_classifier.predict(X_test)

# Step 5: Evaluate accuracy, confusion matrix, and classification report
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

