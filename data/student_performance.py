import pandas as pd

# Load dataset
data = pd.read_csv("data/students.csv")

# Display basic info
print("First 5 rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())
