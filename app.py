import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load Trained Model
MODEL_PATH = "random_forest_model_.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
else:
    print(f"⚠️ Warning: Model file '{MODEL_PATH}' not found in the directory.")

# Feature Mapping Configuration for Human-Friendly Dropdowns
DROPDOWN_OPTIONS = {
    'Make': {'Toyota': 0, 'Honda': 1, 'Ford': 2, 'BMW': 3, 'Mercedes': 4, 'Audi': 5, 'Chevrolet': 6},
    'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'Hybrid': 2, 'Electric': 3},
    'Transmission': {'Manual': 0, 'Automatic': 1, 'Semi-Automatic': 2},
    'Accident_History': {'No Accidents': 0, 'Minor Damage': 1, 'Major Accident': 2},
    'Service_History': {'Full Service History': 1, 'Partial History': 0},
    'Color': {'Black': 0, 'White': 1, 'Silver': 2, 'Blue': 3, 'Red': 4},
    'Body_Type': {'Sedan': 0, 'SUV': 1, 'Hatchback': 2, 'Coupe': 3, 'Convertible': 4},
    'Drivetrain': {'FWD': 0, 'RWD': 1, 'AWD': 2, '4WD': 3},
    'Location': {'Urban Center': 0, 'Suburban': 1, 'Rural': 2}
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoValuate AI | Next-Gen Price Intelligence</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <script src="https://kit.fontawesome.com/a076d05399.js" crossorigin="anonymous"></script>
    <style>
        :root {
            --bg-color: #080c14;
            --panel-bg: rgba(15, 23, 42, 0.75);
            --primary: #6366f1;
            --primary-glow: #818cf8;
            --accent: #06b6d4;
            --accent-glow: rgba(6, 182, 212, 0.4);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: rgba(255, 255, 255, 0.08);
            --input-bg: rgba(2, 6, 23, 0.6);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 3rem 1.5rem;
            position: relative;
            overflow-x: hidden;
        }

        /* Ambient Animated Background Orbs */
        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(120px);
            z-index: 0;
            pointer-events: none;
            animation: float 12s infinite alternate ease-in-out;
        }

        .orb-1 {
            width: 450px;
            height: 450px;
            background: rgba(99, 102, 241, 0.25);
            top: -100px;
            left: -100px;
        }

        .orb-2 {
            width: 500px;
            height: 500px;
            background: rgba(6, 182, 212, 0.2);
            bottom: -150px;
            right: -100px;
            animation-delay: -6s;
        }

        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(50px, 40px) scale(1.15); }
        }

        /* Container Layout */
        .wrapper {
            width: 100%;
            max-width: 1100px;
            z-index: 1;
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 2rem;
        }

        @media (max-width: 968px) {
            .wrapper { grid-template-columns: 1fr; }
        }

        .main-card {
            background: var(--panel-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 28px;
            padding: 2.5rem;
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.6);
        }

        /* Typography & Header */
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

        .header {
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.25rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.85rem;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(129, 140, 248, 0.3);
            border-radius: 50px;
            color: var(--primary-glow);
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 30%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }

        /* Form Grid Structure */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 0.8rem 1rem;
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            color: var(--text-primary);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
            background: rgba(15, 23, 42, 0.9);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 1.1rem;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--primary) 0%, #4338ca 100%);
            color: #ffffff;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px -5px rgba(99, 102, 241, 0.6);
            background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
        }

        /* Floating Sidebar Card for Real-time Results */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .result-panel {
            background: var(--panel-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 28px;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.6);
            min-height: 280px;
            position: relative;
            overflow: hidden;
        }

        .result-panel::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary), var(--accent));
        }

        .val-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .val-amount {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.8rem;
            font-weight: 700;
            color: #ffffff;
            text-shadow: 0 0 20px var(--accent-glow);
            margin: 0.5rem 0;
            transition: all 0.4s ease;
        }

        .model-status {
            font-size: 0.8rem;
            color: #10b981;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            margin-top: 1rem;
            padding: 0.3rem 0.75rem;
            background: rgba(16, 185, 129, 0.1);
            border-radius: 20px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 8px #10b981;
        }

        .spinner {
            display: none;
            width: 32px;
            height: 32px;
            border: 3px solid rgba(255,255,255,0.1);
            border-radius: 50%;
            border-top-color: var(--accent);
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>

    <div class="wrapper">
        <!-- Input Form Section -->
        <div class="main-card">
            <div class="header">
                <span class="badge">🤖 ML Model Integrated</span>
                <h1>Vehicle Price Predictor</h1>
                <p>Configure parameters below to compute an AI-driven valuation.</p>
            </div>

            <form id="predictionForm" class="form-grid">
                
                <!-- Category Inputs with Human Labels -->
                <div class="input-group">
                    <label>Make</label>
                    <select name="Make">
                        {% for key in dropdowns.Make %}
                            <option value="{{ dropdowns.Make[key] }}">{{ key }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="input-group">
                    <label>Model ID</label>
                    <input type="number" name="Model" value="1" min="0" required>
                </div>

                <div class="input-group">
                    <label>Year</label>
                    <input type="number" name="Year" value="2021" min="1990" max="2026" required>
                </div>

                <div class="input-group">
                    <label>Fuel Type</label>
                    <select name="Fuel_Type">
                        {% for key in dropdowns.Fuel_Type %}
                            <option value="{{ dropdowns.Fuel_Type[key] }}">{{ key }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="input-group">
                    <label>Transmission</label>
                    <select name="Transmission">
                        {% for key in dropdowns.Transmission %}
                            <option value="{{ dropdowns.Transmission[key] }}">{{ key }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="input-group">
                    <label>Engine Size (L)</label>
                    <input type="number" step="0.1" name="Engine_Size" value="2.0" required>
                </div>

                <div class="input-group">
                    <label>Mileage (Miles/KM)</label>
                    <input type="number" name="Mileage" value="35000" required>
                </div>

                <div class="input-group">
                    <label>Horsepower (HP)</label>
                    <input type="number" name="Horsepower" value="220" required>
                </div>

                <div class="input-group">
                    <label>Torque (Nm)</label>
                    <input type="number" name="Torque" value="300" required>
                </div>

                <div class="input-group">
                    <label>Previous Owners</label>
                    <input type="number" name="Owners" value="1" min="0" required>
                </div>

                <div class="input-group">
                    <label>Accident History</label>
                    <select name="Accident_History">
                        {% for key in dropdowns.Accident_History %}
                            <option value="{{ dropdowns.Accident_History[key] }}">{{ key }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="input-group">
                    <label>Service History</label>
                    <select name="Service_History">
                        {% for key in dropdowns.Service_History %}
                            <option value="{{ dropdowns.Service_History[key] }}">{{ key }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="input-group">
                    <label>Color</label>
                    <select name="Color">
                        {% for key in dropdowns.Color %}
                            <option value="{{ dropdowns.Color[key] }}">{{ key }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="input-group">
                    <label>Body Type</label>
                    <select name="Body_Type">
                        {% for key in dropdowns.Body_Type %}
                            <option value="{{ dropdowns.Body_Type[key] }}">{{ key }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="input-group">
                    <label>Drivetrain</label>
                    <select name="Drivetrain">
                        {% for key in dropdowns.Drivetrain %}
                            <option value="{{ dropdowns.Drivetrain[key] }}">{{ key }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="input-group">
                    <label>Fuel Efficiency (MPG/KML)</label>
                    <input type="number" step="0.1" name="Fuel_Efficiency" value="28.5" required>
                </div>

                <div class="input-group">
                    <label>Location Region</label>
                    <select name="Location">
                        {% for key in dropdowns.Location %}
                            <option value="{{ dropdowns.Location[key] }}">{{ key }}</option>
                        {% endfor %}
                    </select>
                </div>

                <button type="submit" class="btn-submit" id="submitBtn">
                    <span>Calculate Market Value</span>
                </button>
            </form>
        </div>

        <!-- Sidebar / Result Display Card -->
        <div class="sidebar">
            <div class="result-panel">
                <div class="spinner" id="loadingSpinner"></div>
                
                <div id="resultBox">
                    <div class="val-title">Estimated Market Price</div>
                    <div class="val-amount" id="predictedPrice">$0.00</div>
                </div>

                <div class="model-status">
                    <span class="status-dot"></span>
                    RandomForestRegressor Active
                </div>
            </div>
        </div>
    </div>

    <!-- Async Form Handling Script -->
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = document.getElementById('submitBtn');
            const spinner = document.getElementById('loadingSpinner');
            const priceDisplay = document.getElementById('predictedPrice');
            const formData = new FormData(e.target);

            // UI Feedback: Loading
            spinner.style.display = 'block';
            priceDisplay.style.opacity = '0.3';
            submitBtn.style.pointerEvents = 'none';

            try {
                const response = await fetch('/predict_api', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    priceDisplay.innerText = result.prediction;
                } else {
                    priceDisplay.innerText = "Error";
                    alert(result.error || "An error occurred standardizing inputs.");
                }
            } catch (err) {
                console.error(err);
                priceDisplay.innerText = "Error";
            } finally {
                // UI Reset
                spinner.style.display = 'none';
                priceDisplay.style.opacity = '1';
                submitBtn.style.pointerEvents = 'all';
            }
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, dropdowns=DROPDOWN_OPTIONS)

@app.route("/predict_api", methods=["POST"])
def predict_api():
    """Asynchronous API endpoint to serve requests seamlessly without page reload."""
    if model is None:
        return jsonify({"success": False, "error": "Model file random_forest_model_.pkl not found on server."})

    try:
        # Exact sequence expected by the Random Forest model features
        feature_order = [
            'Make', 'Model', 'Year', 'Fuel_Type', 'Transmission', 'Engine_Size',
            'Mileage', 'Horsepower', 'Torque', 'Owners', 'Accident_History',
            'Service_History', 'Color', 'Body_Type', 'Drivetrain', 'Fuel_Efficiency', 'Location'
        ]

        # Read form features dynamically in correct feature order
        features = [float(request.form.get(feat, 0)) for feat in feature_order]
        features_array = np.array([features])

        # Prediction
        prediction_val = model.predict(features_array)[0]
        formatted_val = f"${prediction_val:,.2f}"

        return jsonify({"success": True, "prediction": formatted_val})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
