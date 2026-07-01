import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Loading data
df = pd.read_csv("data/raw_housing.csv")

# We select the columns we want.
features = ["GrLivArea", "OverallQual", "TotalBsmtSF", "YearBuilt", "BedroomAbvGr", "Neighborhood", "HouseStyle"]
target = "SalePrice"

# Converting character data to numbers (Encoding)
encoders = {}
for col in ["Neighborhood", "HouseStyle"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Fill missing values ​​if there are any blanks.
df[features] = df[features].fillna(0)
df[target] = df[target].fillna(df[target].mean())

# Data splitting (Train and Test)
X = df[features]
y = df[target]

# Scaling the values (Normalize the data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the preprocessed data for future use
joblib.dump(scaler, "data/scaler.pkl")
joblib.dump(encoders, "data/encoders.pkl")
np.save("data/X_train.npy", X_train_scaled)
np.save("data/X_test.npy", X_test_scaled)
np.save("data/y_train.npy", y_train.values)
np.save("data/y_test.npy", y_test.values)

print("Data cleaning and preparation successful!")