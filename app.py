from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and encoders
model = joblib.load("data/best_model.pkl")
scaler = joblib.load("data/scaler.pkl")
encoders = joblib.load("data/encoders.pkl")

@app.route('/')
def home():
    # Pass the dropdown options to the HTML page
    neighborhoods = list(encoders["Neighborhood"].classes_)
    housestyles = list(encoders["HouseStyle"].classes_)
    return render_template('index.html', neighborhoods=neighborhoods, housestyles=housestyles)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        area = float(request.form.get('area'))
        qual = float(request.form.get('qual'))
        neighborhood = request.form.get('neighborhood')
        housestyle = request.form.get('housestyle')
        
        # Encode categorical data
        neigh_val = encoders["Neighborhood"].transform([neighborhood])[0]
        style_val = encoders["HouseStyle"].transform([housestyle])[0]
        
        # Prepare the feature array (7 features)
        # Note: 800, 2000, 3 are placeholder values for TotalBsmtSF, YearBuilt, BedroomAbvGr
        data = np.array([[area, qual, 800, 2000, 3, neigh_val, style_val]])
        scaled_data = scaler.transform(data)
        
        # Predict
        price = model.predict(scaled_data)
        
        return jsonify({'price': f"${price[0]:,.2f}"})
    except Exception as e:
        return jsonify({'price': "Error: Could not process request."})

if __name__ == '__main__':
    app.run(debug=True)