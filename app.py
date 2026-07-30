import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "random_forest_model(3).pkl")

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# Define feature names & categorical options
FEATURE_NAMES = [
    "Make", "Model", "Year", "Fuel_Type", "Transmission", "Engine_Size",
    "Mileage", "Horsepower", "Torque", "Owners", "Accident_History",
    "Service_History", "Color", "Body_Type", "Drivetrain",
    "Fuel_Efficiency", "Location"
]

CATEGORICAL_FEATURES = {
    "Make": ["Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes", "Audi", "Nissan"],
    "Model": ["Sedan", "SUV", "Coupe", "Hatchback", "Truck", "Convertible"],
    "Fuel_Type": ["Petrol", "Diesel", "Hybrid", "Electric"],
    "Transmission": ["Manual", "Automatic", "CVT", "Semi-Automatic"],
    "Accident_History": ["None", "Minor", "Major"],
    "Service_History": ["Full", "Partial", "None"],
    "Color": ["Black", "White", "Silver", "Red", "Blue", "Grey"],
    "Body_Type": ["Sedan", "SUV", "Hatchback", "Coupe", "Truck"],
    "Drivetrain": ["FWD", "RWD", "AWD", "4WD"],
    "Location": ["Urban", "Suburban", "Rural"]
}

# Embedded HTML Template with Modern CSS Styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vehicle Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --input-bg: #0f172a;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border-color);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            padding: 2.5rem 2rem;
            text-align: center;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        }

        .header p {
            color: #e0e7ff;
            font-size: 0.95rem;
        }

        form {
            padding: 2rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 0.75rem 1rem;
            background-color: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-main);
            font-size: 0.95rem;
            transition: all 0.2s ease;
            outline: none;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        .submit-btn {
            width: 100%;
            padding: 1rem;
            background-color: var(--accent-color);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s ease, transform 0.1s ease;
        }

        .submit-btn:hover {
            background-color: var(--accent-hover);
        }

        .submit-btn:active {
            transform: scale(0.99);
        }

        .result-box {
            margin-top: 1.5rem;
            padding: 1.25rem;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid var(--accent-color);
            border-radius: 8px;
            text-align: center;
        }

        .result-box h2 {
            font-size: 1.25rem;
            color: var(--text-main);
        }

        .result-box span {
            color: #818cf8;
            font-weight: 700;
        }

        .error-box {
            margin-top: 1.5rem;
            padding: 1rem;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid #ef4444;
            color: #fca5a5;
            border-radius: 8px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Vehicle Value Predictor</h1>
            <p>Enter vehicle specs below to estimate market value using ML</p>
        </div>
        
        <form action="/predict" method="POST">
            <div class="form-grid">
                {% for feature in features %}
                <div class="input-group">
                    <label for="{{ feature }}">{{ feature.replace('_', ' ') }}</label>
                    {% if feature in categorical_opts %}
                        <select name="{{ feature }}" id="{{ feature }}" required>
                            {% for option in categorical_opts[feature] %}
                                <option value="{{ option }}" {% if form_data and form_data.get(feature) == option %}selected{% endif %}>
                                    {{ option }}
                                </option>
                            {% endfor %}
                        </select>
                    {% else %}
                        <input type="number" step="any" name="{{ feature }}" id="{{ feature }}" 
                               placeholder="Enter {{ feature.replace('_', ' ') }}" 
                               value="{{ form_data.get(feature, '') if form_data else '' }}" required>
                    {% endif %}
                </div>
                {% endfor %}
            </div>

            <button type="submit" class="submit-btn">Predict Value</button>

            {% if prediction %}
            <div class="result-box">
                <h2>Estimated Prediction: <span>{{ prediction }}</span></h2>
            </div>
            {% endif %}

            {% if error %}
            <div class="error-box">
                <p>{{ error }}</p>
            </div>
            {% endif %}
        </form>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML_TEMPLATE, 
        features=FEATURE_NAMES, 
        categorical_opts=CATEGORICAL_FEATURES
    )

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE,
            features=FEATURE_NAMES,
            categorical_opts=CATEGORICAL_FEATURES,
            error="Model file 'random_forest_model.pkl' not found."
        )

    try:
        raw_data = request.form.to_dict()
        input_data = []

        # Parse inputs in order expected by the model
        for feature in FEATURE_NAMES:
            val = raw_data.get(feature)
            if feature in CATEGORICAL_FEATURES:
                # Convert categorical strings into numerical categories if needed by model,
                # or pass directly if model uses string handling
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = hash(val) % 1000  # Fallback code for non-encoded string categories
            else:
                val = float(val)
            input_data.append(val)

        # Make prediction
        features_array = np.array([input_data])
        prediction_val = model.predict(features_array)[0]
        formatted_prediction = f"${prediction_val:,.2f}"

        return render_template_string(
            HTML_TEMPLATE,
            features=FEATURE_NAMES,
            categorical_opts=CATEGORICAL_FEATURES,
            prediction=formatted_prediction,
            form_data=raw_data
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            features=FEATURE_NAMES,
            categorical_opts=CATEGORICAL_FEATURES,
            error=f"Error in prediction: {str(e)}",
            form_data=request.form.to_dict()
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
