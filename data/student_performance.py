import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Load dataset
data = pd.read_csv("data/students.csv")

# Features and target
X = data.drop("pass", axis=1)
y = data["pass"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)

# Precision & Recall
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)

print("\nPrecision:", precision)
print("Recall:", recall)

# Actual vs Predicted
print("\nActual vs Predicted:")
for actual, predicted in zip(y_test.values, predictions):
    print(f"Actual: {actual}, Predicted: {predicted}")
    
# Feature importance (Logistic Regression coefficients)
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

# Sort by absolute importance
feature_importance["Abs_Coefficient"] = feature_importance["Coefficient"].abs()
feature_importance = feature_importance.sort_values(
    by="Abs_Coefficient", ascending=False
)

print("\nFeature Importance:")
print(feature_importance[["Feature", "Coefficient"]])
