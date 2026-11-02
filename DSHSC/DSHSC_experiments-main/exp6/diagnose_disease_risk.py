import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load patient medical dataset (works from any working directory)
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes.csv")
df = pd.read_csv(dataset_path)

# Map target Outcome (0 -> Low Risk, 1 -> High Risk)
df["Risk"] = df["Outcome"].map({0: "Low", 1: "High"})

print("Patient Dataset Sample:")
print(df[["Age", "Glucose", "BloodPressure", "BMI", "Risk"]].head())

# Step 2: Separate features and target
features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
X = df[features]
y = df["Risk"]

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Step 4: Train Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict disease risk for test patients
y_pred = model.predict(X_test)

print("\nSample Predicted Disease Risks (First 5 Test Patients):")
for i, risk in enumerate(y_pred[:5], start=1):
    print(f"Patient {i}: {risk} Risk")

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["Low", "High"]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=["Low", "High"], zero_division=0))

# Step 7: Predict risk for a new patient
new_patient = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 160,
    "BloodPressure": 80,
    "SkinThickness": 35,
    "Insulin": 180,
    "BMI": 32.5,
    "DiabetesPedigreeFunction": 0.65,
    "Age": 48
}])

prediction = model.predict(new_patient)[0]

print("\nNew Patient Diagnostic Check:")
print(new_patient)
print(f"\nPredicted Disease Risk: {prediction} Risk")
