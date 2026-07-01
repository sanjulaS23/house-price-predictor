import numpy as np
import joblib

# We will retrieve the previously saved Model and Scaler.
model = joblib.load("data/best_model.pkl")
scaler = joblib.load("data/scaler.pkl")
encoders = joblib.load("data/encoders.pkl")


# Providing accurate values
target_neighborhood = "CollgCr"
target_housestyle = "1Story"

# Encoding (converting words to numbers)
neighborhood_val = encoders["Neighborhood"].transform([target_neighborhood])[0]
housestyle_val = encoders["HouseStyle"].transform([target_housestyle])[0]

# We will use the model to predict the price of a new house.
# [GrLivArea, OverallQual, TotalBsmtSF, YearBuilt, BedroomAbvGr]
new_house = np.array([[1500, 7, 800, 2000, 3, neighborhood_val, housestyle_val]])

# Scale the new house data using the fitted scaler
new_house_scaled = scaler.transform(new_house)

# We will use the trained model to make a prediction
predicted_price = model.predict(new_house_scaled)

print(f"Price according to Home location ({target_neighborhood}) and Type ({target_housestyle}) ")
print(f"Estimated price of this house: ${predicted_price[0]:,.2f}")