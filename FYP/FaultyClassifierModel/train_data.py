# train_model.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report  # Add these imports
import joblib
from load_data import load_machine_data

def train_and_save_model():
    df = load_machine_data()

    # Encode categorical variables
    df['equipment'] = LabelEncoder().fit_transform(df['equipment'])
    df['location'] = LabelEncoder().fit_transform(df['location'])

    X = df.drop('faulty', axis=1)
    y = df['faulty']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = RandomForestClassifier()
    
    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)  # Predictions on the test set
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")  # Print accuracy
    print("Classification Report:")
    print(classification_report(y_test, y_pred))  # Print detailed classification metrics

    # Save the trained model to a file
    joblib.dump(model, 'machine_fault_model.pkl')
    print("✅ Model trained and saved as machine_fault_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
