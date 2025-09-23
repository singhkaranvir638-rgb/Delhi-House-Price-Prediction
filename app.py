from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract input features from form
            bhk = int(request.form['BHK'])
            bathroom = int(request.form['Bathroom'])
            parking = int(request.form['Parking'])
            per_sqft = float(request.form['Per_Sqft'])
            income = float(request.form['Income'])

            # Create feature array and scale
            features = np.array([[bhk, bathroom, parking, per_sqft, income]])
            features_scaled = scaler.transform(features)

            # Predict price
            pred_price = model.predict(features_scaled)[0]
            prediction = f"₹{pred_price:,.2f}"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
