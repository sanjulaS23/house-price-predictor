import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# We load the data we created earlier.
X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

# Building a Machine Learning Model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Training the Model
print("Training the model...")
model.fit(X_train, y_train)

# Checking the model's talent
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

print(f"Training completed! Model's performance (R² score): {accuracy:.4f}")

# Saving the trained model
joblib.dump(model, "data/best_model.pkl")
print("Model saved as 'data/best_model.pkl'.")