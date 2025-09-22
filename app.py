from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "Welcome to the House Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input JSON data
    data = request.get_json()
    
    # Extract features from the incoming request (ensure this matches the dataset features)
    features = np.array([data['feature1'], data['feature2'], data['feature3']])  # Replace with actual feature names
    
    # Scale the features using the scaler
    features_scaled = scaler.transform([features])
    
    # Make prediction using the model
    prediction = model.predict(features_scaled)
    
    # Return the predicted price as a response
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
