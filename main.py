import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


data_path = 'D:/Unified-Mentor-Internship/Projects/Heart_Disease/dataset/dataset.csv'
explanation_path = 'D:/Unified-Mentor-Internship/Projects/Heart_Disease/dataset/Dataset-explanation.csv'

# Load the dataset and explanation
try:
    data = pd.read_csv(data_path)
    explanation = pd.read_csv(explanation_path)
    print("Dataset and explanation loaded successfully.")
except Exception as e:
    print("Error loading datasets:", e)
    exit()

# Explore the dataset
print("\nFirst few rows of the dataset:")
print(data.head())

print("\nDataset information:")
data.info()

print("\nChecking for missing values:")
print(data.isnull().sum())

# Basic preprocessing: Handle missing values if any
if data.isnull().values.any():
    data = data.fillna(data.mean())  # Filling missing values with mean as a simple strategy
    print("Missing values handled.")

# Feature-target split
target_column = 'target'  # Adjust this based on the dataset explanation
X = data.drop(columns=[target_column])
y = data[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Use a RandomForestClassifier for prediction
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy of the model:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the preprocessed data and model if needed
processed_data_path = 'D:/Unified-Mentor-Internship/Projects/Heart_Disease/dataset/processed_data.csv'
data.to_csv(processed_data_path, index=False)
print(f"Preprocessed data saved to {processed_data_path}")
