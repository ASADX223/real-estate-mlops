
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download the real housing dataset
print("Downloading California Housing dataset...")

housing = fetch_california_housing(as_frame=True)

# Combine features and target into one DataFrame
df = housing.frame.copy()

# Rename the target to make its meaning clear
df = df.rename(columns={"MedHouseVal": "median_house_value_100k"})

# Save the dataset
output_path = DATA_DIR / "housing.csv"
df.to_csv(output_path, index=False)

print("\nDataset prepared successfully!")
print(f"Saved to: {output_path}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())
