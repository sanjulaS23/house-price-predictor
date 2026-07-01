import pandas as pd
from sklearn.datasets import fetch_openml
import os

# Creating a folder to keep data safe
os.makedirs("data", exist_ok=True)

print("Downloading data... (Please wait)")
housing = fetch_openml(name="house_prices", as_frame=True)
df = housing.frame

# Shows the amount of data
print(f"Success! Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

# Showing the first 5 rows
print(df.head())

# Save the data as a csv file
df.to_csv("data/raw_housing.csv", index=False)
print("Data saved as 'data/raw_housing.csv'.")