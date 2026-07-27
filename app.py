import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the Scikit-learn Random Forest model
MODEL_PATH = "random_forest_model_.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# HTML Template with Embedded CSS & JS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-grad: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --accent-grad: linear-gradient(135deg, #3b82f6 0%, #2dd4bf 100%);
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text-color: #f8fafc;
            --input-bg: rgba(15, 23, 42, 0.6);
            --border-color: rgba(255, 255, 255, 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Poppins', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow-x: hidden;
            padding: 40px 20px;
            position: relative;
        }

        /* Animated Background Orbs */
        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            z-index: 0;
            animation: float 10s infinite ease-in-out alternate;
        }

        .orb-1 {
            width: 300px;
            height: 300px;
            background: rgba(99, 102, 241, 0.3);
            top: 10%;
            left: 10%;
        }

        .orb-2 {
            width: 350px;
            height: 350px;
            background: rgba(168, 85, 247, 0.25);
            bottom: 10%;
            right: 10%;
            animation-delay: -5s;
        }

        @keyframes float {
            0% { transform: translateY(0) scale(1); }
            100% { transform: translateY(-30px) scale(1.1); }
        }

        /* Container Card */
        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1 {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            background: var(--primary-grad);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        p.subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 0.95rem;
            margin-bottom: 30px;
        }

        /* Form Grid */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: #cbd5e1;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .input-group input {
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            color: #fff;
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus {
            border-color: #a855f7;
            box-shadow: 0 0 12px rgba(168, 85, 247, 0.3);
            background: rgba(15, 23, 42, 0.8);
        }

        /* Button Styling */
        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 15px;
            padding: 16px;
            font-size: 1.1rem;
            font-weight: 600;
            color: #fff;
            background: var(--primary-grad);
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.3s ease;
            box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px rgba(168, 85, 247, 0.4);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* Result Display Box */
        .result-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 16px;
            background: var(--accent-grad);
            text-align: center;
            display: none;
            animation: pulseIn 0.5s ease-out forwards;
        }

        @keyframes pulseIn {
            0% { opacity: 0; transform: scale(0.9); }
            100% { opacity: 1; transform: scale(1); }
        }

        .result-box h2 {
            font-size: 1.2rem;
            color: #f8fafc;
            font-weight: 400;
        }

        .result-box span {
            font-size: 2.2rem;
            font-weight: 700;
            display: block;
            margin-top: 5px;
        }
    </style>
</head>
<body>

    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>

    <div class="container">
        <h1>Car Price Predictor</h1>
        <p class="subtitle">Enter the numeric feature values to estimate the car value</p>

        <form id="predictionForm">
            <div class="form-grid">
                <div class="input-group"><label>Make</label><input type="number" step="any" name="Make" required value="0"></div>
                <div class="input-group"><label>Model</label><input type="number" step="any" name="Model" required value="0"></div>
                <div class="input-group"><label>Year</label><input type="number" name="Year" required value="2020"></div>
                <div class="input-group"><label>Fuel Type</label><input type="number" step="any" name="Fuel_Type" required value="0"></div>
                <div class="input-group"><label>Transmission</label><input type="number" step="any" name="Transmission" required value="0"></div>
                <div class="input-group"><label>Engine Size</label><input type="number" step="any" name="Engine_Size" required value="2.0"></div>
                <div class="input-group"><label>Mileage</label><input type="number" step="any" name="Mileage" required value="45000"></div>
                <div class="input-group"><label>Horsepower</label><input type="number" step="any" name="Horsepower" required value="180"></div>
                <div class="input-group"><label>Torque</label><input type="number" step="any" name="Torque" required value="250"></div>
                <div class="input-group"><label>Owners</label><input type="number" name="Owners" required value="1"></div>
                <div class="input-group"><label>Accident History</label><input type="number" step="any" name="Accident_History" required value="0"></div>
                <div class="input-group"><label>Service History</label><input type="number" step="any" name="Service_History" required value="1"></div>
                <div class="input-group"><label>Color</label><input type="number" step="any" name="Color" required value="0"></div>
                <div class="input-group"><label>Body Type</label><input type="number" step="any" name="Body_Type" required value="0"></div>
                <div class="input-group"><label>Drivetrain</label><input type="number" step="any" name="Drivetrain" required value="0"></div>
                <div class="input-group"><label>Fuel Efficiency</label><input type="number" step="any" name="Fuel_Efficiency" required value="15.5"></div>
                <div class="input-group"><label>Location</label><input type="number" step="any" name="Location" required value="0"></div>

                <button type="submit" class="btn-submit">Predict Value</button>
            </div>
        </form>

        <div id="resultBox" class="result-box">
            <h2>Estimated Car Price</h2>
            <span id="predictedPrice">$0.00</span>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => { data[key] = parseFloat(value); });

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            const resultBox = document.getElementById('resultBox');
            const priceSpan = document.getElementById('predictedPrice');

            if (result.success) {
                priceSpan.innerText = '$' + result.prediction.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                resultBox.style.display = 'block';
            } else {
                alert('Prediction Error: ' + result.error);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model file not loaded properly on the server.'})

    try:
        data = request.get_json()
        
        # Order inputs strictly matching feature_names_in_ from the trained model
        feature_order = [
            'Make', 'Model', 'Year', 'Fuel_Type', 'Transmission', 'Engine_Size', 
            'Mileage', 'Horsepower', 'Torque', 'Owners', 'Accident_History', 
            'Service_History', 'Color', 'Body_Type', 'Drivetrain', 
            'Fuel_Efficiency', 'Location'
        ]
        
        features = [float(data.get(feat, 0)) for feat in feature_order]
        prediction = model.predict([features])[0]

        return jsonify({'success': True, 'prediction': float(prediction)})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
