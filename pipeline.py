import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/raw/breast_cancer.csv")

df = df.dropna()
X = df.drop(columns=["target"])
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

processed = pd.DataFrame(X_scaled, columns=X.columns)
processed["target"] = y.values

processed.to_csv("data/processed/breast_cancer_processed.csv", index=False)

print("Pipeline completed successfully!")
print("Processed shape:", processed.shape)