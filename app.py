from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('gradient_boosting_regressor.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)
    
    # Ensure that the data is in the correct format (e.g., a list of lists for multiple samples)
    # This part may need to be customized based on how the data needs to be processed
    prediction_features = [data['features']]
    
    # Make prediction
    prediction = model.predict(prediction_features)
    
    # Return the prediction as a JSON object
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
