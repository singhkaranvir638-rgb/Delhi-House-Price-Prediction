from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('house_price_model.pkl')
# If you also saved the scaler, load it here:
# scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        sqft_living = float(request.form['sqft_living'])
        sqft_lot = float(request.form['sqft_lot'])
        floors = float(request.form['floors'])
        waterfront = float(request.form['waterfront'])
        view = float(request.form['view'])
        condition = float(request.form['condition'])
        sqft_above = float(request.form['sqft_above'])
        sqft_basement = float(request.form['sqft_basement'])
        yr_built = int(request.form['yr_built'])
        yr_renovated = int(request.form['yr_renovated'])
        
        # Create the input array
        input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition,
                                sqft_above, sqft_basement, yr_built, yr_renovated]])

        # Scale the input data if you have a scaler
        # input_data_scaled = scaler.transform(input_data)

        # Get the prediction
        predicted_price = model.predict(input_data)  # Use scaled data if you have a scaler

        # Return the prediction as a response
        return render_template('index.html', prediction_text=f"Predicted Price: ₹{predicted_price[0]:,.2f}")
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Route for Google Colab link
@app.route('/colab')
def colab():
    # Link to your Google Colab notebook
    colab_url = "https://colab.research.google.com/drive/1Rdzpe0RaYJ2C6Xa7eAsP4EPFydgtMNBH"  # Replace this with your actual link
    return render_template('colab.html', colab_link=colab_url)

if __name__ == '__main__':
    app.run(debug=True)
