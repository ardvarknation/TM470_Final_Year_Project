# Additional imports for Random Forest Classifiers and associated metrics.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import time     # For training timing as measure of complexity and computational cost.

# Read in data from CSV file.
print("Loading data...")
dataset = pd.read_csv("/content/.../labelled_bsl_data.csv")
print("Data loaded.")

# Separate the features and labels from (features, label) pairings in CSV file.
X = dataset.iloc[:, :-1]  # Select all rows and columns except the last (:-1) and assigns as features (X).
y = dataset.iloc[:, -1]   # Select all rows and only last column (-1) and assign as class label (y).

# Produce training and testing splits.
print("Splitting into training and testing datasets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Splitting complete.")

# Attempt to standardise features prior to training for consistent results.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier with specified number of estimators.
print("\nTraining model...")
# Comment out line below when uncommenting models 2 or 3.
rf_model_1 = RandomForestClassifier(n_estimators=33)  # Equal number of estimators as classes.

# Uncomment line below for second iteration of model with larger number of estimators.
# rf_model_2 = RandomForestClassifier(n_estimators=100)

# Uncomment line below for third iteration of model with largest number of estimators.
# rf_model_3 = RandomForestClassifier(n_estimators=200)

print(rf_model_1)    # Change to rf_model_2 or rf_model_3 as required.

# Start timer
start = time.time()

rf_model_1.fit(X_train, y_train)  # Change to rf_model_2.fit() or rf_model_3.fit() as required.

# End timer.
end = time.time()

print("Model trained in: " + str(end - start) + " seconds.")

print("\nTesting model...")

# Generate predictions for the Random Forest Classifier from test data.
pred_values = rf_model_1.predict(X_test)  # Change to rf_model_2.predict() or rf_model_3.predict() as per above.

# Store the metrics captured from test data and predicted values.
acc = accuracy_score(y_test, pred_values)
prec = precision_score(y_test, pred_values, average='weighted')
rec = recall_score(y_test, pred_values, average='weighted')
f1 = f1_score(y_test, pred_values, average='weighted')
report = classification_report(y_test, pred_values)

# Display report of performance.
print("\nValidation metrics:")
print("Accuracy: " + str(acc*100) + "%")
print("Precision: " + str(prec))
print("Recall: " + str(rec))
print("F1 Score: " + str(f1))
print("\nClassification report:")
print(report)
