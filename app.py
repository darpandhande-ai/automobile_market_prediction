import os
import pickle
import pandas as pd
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# Load Model
MODEL_PATH = "random_forest_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")

# Embedded HTML Template with Glassmorphism CSS Styles
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Price Predictor | Machine Learning Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --input-border: rgba(255, 255, 255, 0.15);
            --accent-green: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background: var(--bg-gradient);
            background-attachment: fixed;
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 40px 20px;
        }

        .container {
            width: 100%;
            max-width: 950px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 35px;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #818cf8, #c084fc, #e879f9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .grid-layout {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .form-control {
            width: 100%;
            padding: 12px 16px;
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 12px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
        }

        .form-control:focus {
            border-color: #818cf8;
            box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.25);
        }

        select.form-control {
            appearance: none;
            background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
            background-repeat: no-repeat;
            background-position: right 14px center;
            background-size: 16px;
            cursor: pointer;
        }

        select.form-control option {
            background-color: #1e293b;
            color: var(--text-main);
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 15px;
            padding: 16px;
            background: linear-gradient(135deg, var(--primary) 0%, #6366f1 100%);
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 600;
            border: none;
            border-radius: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px -5px rgba(79, 70, 229, 0.4);
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(79, 70, 229, 0.6);
            background: linear-gradient(135deg, #4338ca 0%, var(--primary) 100%);
        }

        .result-card {
            margin-top: 35px;
            padding: 24px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 16px;
            text-align: center;
            animation: fadeIn 0.4s ease-out;
        }

        .result-card h3 {
            color: var(--text-muted);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }

        .result-card .price {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--accent-green);
        }

        .error-card {
            margin-top: 35px;
            padding: 20px;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 16px;
            color: #f87171;
            text-align: center;
            font-weight: 500;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 640px) {
            .container { padding: 25px 20px; }
            .header h1 { font-size: 1.75rem; }
            .grid-layout { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Car Price Prediction Portal</h1>
        <p>Enter vehicle specifications below to calculate estimated market value</p>
    </div>

    <form action="/predict" method="POST">
        <div class="grid-layout">
            <div class="form-group">
                <label for="Make">Make</label>
                <input type="text" id="Make" name="Make" class="form-control" placeholder="e.g. Toyota" value="{{ request.form.get('Make', '') }}" required>
            </div>

            <div class="form-group">
                <label for="Model">Model</label>
                <input type="text" id="Model" name="Model" class="form-control" placeholder="e.g. Camry" value="{{ request.form.get('Model', '') }}" required>
            </div>

            <div class="form-group">
                <label for="Year">Year</label>
                <input type="number" id="Year" name="Year" class="form-control" min="1990" max="2026" placeholder="2020" value="{{ request.form.get('Year', '') }}" required>
            </div>

            <div class="form-group">
                <label for="Fuel_Type">Fuel Type</label>
                <select id="Fuel_Type" name="Fuel_Type" class="form-control" required>
                    <option value="" disabled {{ 'selected' if not request.form.get('Fuel_Type') }}>Select Fuel</option>
                    {% for option in ['Petrol', 'Diesel', 'Electric', 'Hybrid', 'CNG'] %}
                    <option value="{{ option }}" {{ 'selected' if request.form.get('Fuel_Type') == option }}>{{ option }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="form-group">
                <label for="Transmission">Transmission</label>
                <select id="Transmission" name="Transmission" class="form-control" required>
                    <option value="" disabled {{ 'selected' if not request.form.get('Transmission') }}>Select Transmission</option>
                    {% for option in ['Manual', 'Automatic', 'Semi-Automatic'] %}
                    <option value="{{ option }}" {{ 'selected' if request.form.get('Transmission') == option }}>{{ option }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="form-group">
                <label for="Engine_Size">Engine Size (L)</label>
                <input type="number" step="0.1" id="Engine_Size" name="Engine_Size" class="form-control" placeholder="e.g. 2.0" value="{{ request.form.get('Engine_Size', '') }}" required>
            </div>

            <div class="form-group">
                <label for="Mileage">Mileage (km)</label>
                <input type="number" step="1" id="Mileage" name="Mileage" class="form-control" placeholder="e.g. 45000" value="{{ request.form.get('Mileage', '') }}" required>
            </div>

            <div class="form-group">
                <label for="Horsepower">Horsepower (HP)</label>
                <input type="number" step="1" id="Horsepower" name="Horsepower" class="form-control" placeholder="e.g. 180" value="{{ request.form.get('Horsepower', '') }}" required>
            </div>

            <div class="form-group">
                <label for="Torque">Torque (Nm)</label>
                <input type="number" step="1" id="Torque" name="Torque" class="form-control" placeholder="e.g. 250" value="{{ request.form.get('Torque', '') }}" required>
            </div>

            <div class="form-group">
                <label for="Owners">Previous Owners</label>
                <input type="number" id="Owners" name="Owners" class="form-control" min="0" max="10" placeholder="e.g. 1" value="{{ request.form.get('Owners', '') }}" required>
            </div>

            <div class="form-group">
                <label for="Accident_History">Accident History</label>
                <select id="Accident_History" name="Accident_History" class="form-control" required>
                    {% for option in ['None', 'Minor', 'Major'] %}
                    <option value="{{ option }}" {{ 'selected' if request.form.get('Accident_History') == option }}>{{ option }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="form-group">
                <label for="Service_History">Service History</label>
                <select id="Service_History" name="Service_History" class="form-control" required>
                    {% for option in ['Full', 'Partial', 'None'] %}
                    <option value="{{ option }}" {{ 'selected' if request.form.get('Service_History') == option }}>{{ option }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="form-group">
                <label for="Color">Color</label>
                <input type="text" id="Color" name="Color" class="form-control" placeholder="e.g. Black" value="{{ request.form.get('Color', '') }}" required>
            </div>

            <div class="form-group">
                <label for="Body_Type">Body Type</label>
                <select id="Body_Type" name="Body_Type" class="form-control" required>
                    {% for option in ['Sedan', 'SUV', 'Hatchback', 'Coupe', 'Convertible', 'Wagon'] %}
                    <option value="{{ option }}" {{ 'selected' if request.form.get('Body_Type') == option }}>{{ option }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="form-group">
                <label for="Drivetrain">Drivetrain</label>
                <select id="Drivetrain" name="Drivetrain" class="form-control" required>
                    {% for option in ['FWD', 'RWD', 'AWD', '4WD'] %}
                    <option value="{{ option }}" {{ 'selected' if request.form.get('Drivetrain') == option }}>{{ option }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="form-group">
                <label for="Fuel_Efficiency">Fuel Efficiency (km/L)</label>
                <input type="number" step="0.1" id="Fuel_Efficiency" name="Fuel_Efficiency" class="form-control" placeholder="e.g. 15.5" value="{{ request.form.get('Fuel_Efficiency', '') }}" required>
            </div>

            <div class="form-group">
                <label for="Location">Location</label>
                <input type="text" id="Location" name="Location" class="form-control" placeholder="e.g. Urban" value="{{ request.form.get('Location', '') }}" required>
            </div>

            <button type="submit" class="submit-btn">⚡ Calculate Estimated Price</button>
        </div>
    </form>

    {% if prediction_text %}
    <div class="result-card">
        <h3>Estimated Valuation</h3>
        <div class="price">{{ prediction_text }}</div>
    </div>
    {% endif %}

    {% if error_text %}
    <div class="error-card">
        {{ error_text }}
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE, 
            error_text="Error: Model file ('random_forest_model.pkl') not found or failed to load."
        )

    try:
        input_data = {
            'Make': request.form.get('Make'),
            'Model': request.form.get('Model'),
            'Year': float(request.form.get('Year', 0)),
            'Fuel_Type': request.form.get('Fuel_Type'),
            'Transmission': request.form.get('Transmission'),
            'Engine_Size': float(request.form.get('Engine_Size', 0)),
            'Mileage': float(request.form.get('Mileage', 0)),
            'Horsepower': float(request.form.get('Horsepower', 0)),
            'Torque': float(request.form.get('Torque', 0)),
            'Owners': float(request.form.get('Owners', 0)),
            'Accident_History': request.form.get('Accident_History'),
            'Service_History': request.form.get('Service_History'),
            'Color': request.form.get('Color'),
            'Body_Type': request.form.get('Body_Type'),
            'Drivetrain': request.form.get('Drivetrain'),
            'Fuel_Efficiency': float(request.form.get('Fuel_Efficiency', 0)),
            'Location': request.form.get('Location')
        }

        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
        formatted_price = f"${prediction:,.2f}"

        return render_template_string(HTML_TEMPLATE, prediction_text=formatted_price)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error_text=f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
