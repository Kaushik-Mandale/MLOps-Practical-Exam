
from sklearn.datasets import load_breast_cancer
import os

# Load Wisconsin Breast Cancer dataset
dataset = load_breast_cancer(as_frame=True)

# Convert to DataFrame
df = dataset.frame

# Create dataset directory
os.makedirs("data/raw", exist_ok=True)

# Save dataset
df.to_csv("data/raw/breast_cancer.csv", index=False)

print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)
print("Features:", len(dataset.feature_names))
print("Target classes:", dataset.target_names)
print(df.head())
