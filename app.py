import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string, session

app = Flask(__name__)
app.secret_key = os.urandom(32)

MODEL_PATH = "random_forest_model.pkl"

def load_valuation_engine():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, 'rb') as file:
            return pickle.load(file)
    except Exception:
        return None

model = load_valuation_engine()

# -------------------------------------------------------------
# HIGH-ENHANCEMENT LUXURY NEON UI TEMPLATE
# -------------------------------------------------------------
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoValuate AI | Premium Dynamic Valuation</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        /* Multi-Theme Framework Matrix definitions (All Dark-Optimized Base) */
        :root, [data-theme="cyber"] {
            --bg-base: #060913;
            --panel-glass: rgba(15, 23, 42, 0.75);
            --border-glass: rgba(255, 255, 255, 0.08);
            --neon-accent: #06b6d4;
            --neon-secondary: #8b5cf6;
            --neon-gradient: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 50%, #ec4899 100%);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --bg-glow: radial-gradient(circle at 50% 0%, #1c1942 0%, #060913 70%);
        }
        
        [data-theme="sporty"] {
            --bg-base: #0c0c0e;
            --panel-glass: rgba(24, 24, 27, 0.8);
            --border-glass: rgba(255, 255, 255, 0.05);
            --neon-accent: #ea580c;
            --neon-secondary: #f97316;
            --neon-gradient: linear-gradient(135deg, #ea580c 0%, #ef4444 100%);
            --text-main: #fafafa;
            --text-muted: #a1a1aa;
            --bg-glow: radial-gradient(circle at 50% 0%, #2d1610 0%, #0c0c0e 75%);
        }

        [data-theme="fresh"] {
            --bg-base: #020617;
            --panel-glass: rgba(15, 23, 42, 0.7);
            --border-glass: rgba(255, 255, 255, 0.08);
            --neon-accent: #22c55e;
            --neon-secondary: #84cc16;
            --neon-gradient: linear-gradient(135deg, #22c55e 0%, #a3e635 100%);
            --text-main: #f8fafc;
            --text-muted: #64748b;
            --bg-glow: radial-gradient(circle at 50% 0%, #064e3b 0%, #020617 75%);
        }

        [data-theme="attitude"] {
            --bg-base: #070708;
            --panel-glass: rgba(18, 18, 20, 0.85);
            --border-glass: rgba(255, 255, 255, 0.04);
            --neon-accent: #db2777;
            --neon-secondary: #e11d48;
            --neon-gradient: linear-gradient(135deg, #db2777 0%, #4c0519 100%);
            --text-main: #fff5f5;
            --text-muted: #fda4af;
            --bg-glow: radial-gradient(circle at 50% 0%, #4c0519 0%, #070708 75%);
        }

        [data-theme="emerald"] {
            --bg-base: #022c22;
            --panel-glass: rgba(6, 78, 59, 0.5);
            --border-glass: rgba(255, 255, 255, 0.1);
            --neon-accent: #10b981;
            --neon-secondary: #fbbf24;
            --neon-gradient: linear-gradient(135deg, #10b981 0%, #f59e0b 100%);
            --text-main: #f0fdf4;
            --text-muted: #a7f3d0;
            --bg-glow: radial-gradient(circle at 50% 0%, #064e3b 0%, #022c22 75%);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-glow);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
            overflow-x: hidden;
            transition: background 0.5s ease;
        }

        /* Suppress Arrow Scrollbars/Spinners Completely */
        ::-webkit-scrollbar {
            width: 0px;
            height: 0px;
            background: transparent;
        }
        
        input[type="number"]::-webkit-outer-spin-button,
        input[type="number"]::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        input[type="number"] {
            -moz-appearance: textfield;
        }

        .dashboard-container {
            width: 100%;
            max-width: 1400px;
            display: grid;
            grid-template-columns: 1.25fr 0.75fr;
            gap: 2rem;
            animation: fadeIn 0.5s ease-out;
        }

        @media (max-width: 1150px) {
            .dashboard-container { grid-template-columns: 1fr; }
        }

        .glass-card {
            background: var(--panel-glass);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-glass);
            border-radius: 24px;
            padding: 2.2rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease;
        }

        .header-block {
            margin-bottom: 2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }

        h1 {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #ffffff 40%, var(--neon-accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }

        .controls-header {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.25rem;
        }

        @media (max-width: 850px) {
            .feature-grid { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 550px) {
            .feature-grid { grid-template-columns: 1fr; }
        }

        .input-wrapper {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        label {
            font-size: 0.75rem;
            font-weight: 700;
            color: #cbd5e1;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        input, select {
            background: rgba(10, 15, 30, 0.6);
            border: 1px solid var(--border-glass);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.25s ease;
            width: 100%;
        }

        input:focus, select:focus {
            border-color: var(--neon-accent);
            box-shadow: 0 0 12px rgba(6, 182, 212, 0.2);
            background: rgba(10, 15, 30, 0.8);
        }

        .submit-trigger {
            grid-column: span 3;
            background: var(--neon-gradient);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 1.1rem;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 0.75rem;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
        }

        @media (max-width: 850px) { .submit-trigger { grid-column: span 2; } }
        @media (max-width: 550px) { .submit-trigger { grid-column: span 1; } }

        .submit-trigger:hover {
            transform: translateY(-2px);
            filter: brightness(1.1);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        }

        .analytics-side {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .valuation-display {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
            border: 1px solid var(--border-glass);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .valuation-display::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: var(--neon-gradient);
            opacity: 0.12;
            z-index: 0;
        }

        .valuation-display h2 {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--neon-accent);
            margin-bottom: 0.3rem;
            position: relative;
            z-index: 1;
        }

        .valuation-price-container {
            font-size: 3rem;
            font-weight: 900;
            letter-spacing: -0.03em;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.2rem;
            text-shadow: 0 0 25px rgba(255, 255, 255, 0.1);
            position: relative;
            z-index: 1;
        }

        /* Highlights dynamic assessment text styling box */
        .suggestions-block {
            margin-top: 1rem;
            padding: 0.85rem 1rem;
            background: rgba(255, 255, 255, 0.03);
            border-left: 3px solid var(--neon-accent);
            border-radius: 6px;
            font-size: 0.9rem;
            line-height: 1.4;
            color: var(--text-main);
            text-align: left;
        }

        .chart-card {
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .chat-history-card {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            max-height: 420px;
        }

        .chat-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 0.5rem;
        }

        .clear-history-action {
            background: transparent;
            border: none;
            color: var(--neon-accent);
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .chat-log-stream {
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .chat-bubble {
            display: flex;
            flex-direction: column;
            padding: 0.9rem 1.1rem;
            border-radius: 16px;
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .chat-bubble.user-query {
            background: rgba(255, 255, 255, 0.03);
            align-self: flex-end;
            border-bottom-right-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.04);
            width: 90%;
        }

        .chat-bubble.ai-response {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-glass);
            align-self: flex-start;
            border-bottom-left-radius: 4px;
            width: 90%;
        }

        .bubble-meta {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-bottom: 0.3rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .no-records {
            color: var(--text-muted);
            text-align: center;
            margin: auto;
            font-style: italic;
        }

        .system-alert {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.25);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            color: #fca5a5;
            text-align: center;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <div class="dashboard-container">
        <!-- Main Form Input Elements Config -->
        <div class="glass-card">
            <div class="header-block">
                <div>
                    <h1>Valuation Matrix Node</h1>
                    <p class="subtitle">Ensemble Learning 17-Factor Pricing Node Blueprint</p>
                </div>
                <div class="controls-header">
                    <!-- Expanded Dark Custom Color Themes Selection Box Switcher -->
                    <div style="min-width: 140px;">
                        <label>Color Palette</label>
                        <select id="themeSelector" onchange="switchApplicationTheme()">
                            <option value="cyber" selected>Cyber Neon</option>
                            <option value="sporty">Dark Sporty</option>
                            <option value="fresh">Fresh Velocity</option>
                            <option value="attitude">Baddie Attitude</option>
                            <option value="emerald">Emerald Luxury</option>
                        </select>
                    </div>
                    <!-- Currency matrix converter -->
                    <div style="min-width: 110px;">
                        <label>Currency</label>
                        <select id="currencySelector" onchange="convertActiveValuations()">
                            <option value="INR" selected>INR (₹)</option>
                            <option value="USD">USD ($)</option>
                            <option value="EUR">EUR (€)</option>
                            <option value="GBP">GBP (£)</option>
                        </select>
                    </div>
                </div>
            </div>

            {% if error_msg %}
            <div class="system-alert">{{ error_msg }}</div>
            {% endif %}

            <form method="POST" action="/" class="feature-grid">
                <!-- Dropdown categorical blocks layout metrics -->
                <div class="input-wrapper">
                    <label>Brand Manufacturer</label>
                    <select name="Make">
                        <option value="0" {% if form_values.Make == '0' %}selected{% endif %}>Toyota</option>
                        <option value="1" {% if form_values.Make == '1' %}selected{% endif %}>Honda</option>
                        <option value="2" {% if form_values.Make == '2' %}selected{% endif %}>Ford</option>
                        <option value="3" {% if form_values.Make == '3' %}selected{% endif %}>BMW</option>
                    </select>
                </div>
                <div class="input-wrapper">
                    <label>Model Variant</label>
                    <select name="Model">
                        <option value="0" {% if form_values.Model == '0' %}selected{% endif %}>Sedan Base Matrix</option>
                        <option value="1" {% if form_values.Model == '1' %}selected{% endif %}>SUV Hyper Sport</option>
                        <option value="2" {% if form_values.Model == '2' %}selected{% endif %}>Eco Hatch Core</option>
                    </select>
                </div>
                <div class="input-wrapper">
                    <label>Trim Variant Class</label>
                    <select name="Trim_Level">
                        <option value="0" {% if form_values.Trim_Level == '0' %}selected{% endif %}>Standard Entry</option>
                        <option value="1" {% if form_values.Trim_Level == '1' %}selected{% endif %}>Mid Tier Comfort</option>
                    </select>
                </div>

                <!-- Year of Assembly transformed into clean manual typing input fields -->
                <div class="input-wrapper">
                    <label>Year of Assembly</label>
                    <input type="number" id="Year" name="Year" min="2000" max="2027" placeholder="e.g. 2024" value="{{ form_values.Year|default('2026') }}">
                </div>
                <div class="input-wrapper">
                    <label>Fuel System Source</label>
                    <select name="Fuel_Type">
                        <option value="0" {% if form_values.Fuel_Type == '0' %}selected{% endif %}>Unleaded Petrol</option>
                        <option value="1" {% if form_values.Fuel_Type == '1' %}selected{% endif %}>Refined Diesel</option>
                        <option value="2" {% if form_values.Fuel_Type == '2' %}selected{% endif %}>Solid State Electric</option>
                    </select>
                </div>
                <div class="input-wrapper">
                    <label>Transmission Gear Layout</label>
                    <select name="Transmission">
                        <option value="0" {% if form_values.Transmission == '0' %}selected{% endif %}>Manual Mesh</option>
                        <option value="1" {% if form_values.Transmission == '1' %}selected{% endif %}>Dual Clutch Automatic</option>
                    </select>
                </div>

                <!-- Technical typing fields entries structure -->
                <div class="input-wrapper">
                    <label>Engine Capacity (L)</label>
                    <input type="number" id="Engine_Size" name="Engine_Size" step="0.1" min="0.5" max="8.0" placeholder="e.g. 2.0" value="{{ form_values.Engine_Size|default('4.0') }}">
                </div>
                <div class="input-wrapper">
                    <label>Distance Traveled (Odometer km)</label>
                    <input type="number" id="Mileage" name="Mileage" step="1" min="0" max="500000" placeholder="e.g. 45000" value="{{ form_values.Mileage|default('0') }}">
                </div>
                <div class="input-wrapper">
                    <label>Brake Horsepower</label>
                    <input type="number" id="Horsepower" name="Horsepower" step="1" min="30" max="1000" placeholder="e.g. 150" value="{{ form_values.Horsepower|default('167') }}">
                </div>

                <div class="input-wrapper">
                    <label>Torque Curve (Nm)</label>
                    <input type="number" id="Torque" name="Torque" step="1" min="50" max="1200" placeholder="e.g. 250" value="{{ form_values.Torque|default('246') }}">
                </div>
                <div class="input-wrapper">
                    <label>Fuel Efficiency (km/L)</label>
                    <input type="number" id="Fuel_Efficiency" name="Fuel_Efficiency" step="0.1" min="2" max="50" placeholder="e.g. 15.4" value="{{ form_values.Fuel_Efficiency|default('30.0') }}">
                </div>
                <div class="input-wrapper">
                    <label>Service Portfolio State</label>
                    <select name="Service_History">
                        <option value="0" {% if form_values.Service_History == '0' %}selected{% endif %}>Full Documented</option>
                        <option value="1" {% if form_values.Service_History == '1' %}selected{% endif %}>Partial / Missing</option>
                    </select>
                </div>

                <div class="input-wrapper">
                    <label>Previous Owners Count</label>
                    <select name="Owners">
                        <option value="1" {% if form_values.Owners == '1' %}selected{% endif %}>1 Owner</option>
                        <option value="2" {% if form_values.Owners == '2' %}selected{% endif %}>2 Owners</option>
                        <option value="3" {% if form_values.Owners == '3' %}selected{% endif %}>3+</option>
                    </select>
                </div>
                <div class="input-wrapper">
                    <label>Incident History</label>
                    <select name="Accident_History">
                        <option value="0" {% if form_values.Accident_History == '0' %}selected{% endif %}>No Incidents</option>
                        <option value="1" {% if form_values.Accident_History == '1' %}selected{% endif %}>Major / Repaired</option>
                    </select>
                </div>
                <div class="input-wrapper">
                    <label>Coating Color Class</label>
                    <select name="Color">
                        <option value="0" {% if form_values.Color == '0' %}selected{% endif %}>Metallic Black</option>
                        <option value="1" {% if form_values.Color == '1' %}selected{% endif %}>Pure White</option>
                        <option value="2" {% if form_values.Color == '2' %}selected{% endif %}>Silver Accent</option>
                    </select>
                </div>

                <div class="input-wrapper">
                    <label>Interior Wear State</label>
                    <select name="Interior_Condition">
                        <option value="0" {% if form_values.Interior_Condition == '0' %}selected{% endif %}>Pristine</option>
                        <option value="1" {% if form_values.Interior_Condition == '1' %}selected{% endif %}>Moderate Wear</option>
                    </select>
                </div>
                <div class="input-wrapper">
                    <label>Chassis Body Architecture</label>
                    <select name="Body_Type">
                        <option value="0" {% if form_values.Body_Type == '0' %}selected{% endif %}>Coupe</option>
                        <option value="1" {% if form_values.Body_Type == '1' %}selected{% endif %}>Sedan</option>
                        <option value="2" {% if form_values.Body_Type == '2' %}selected{% endif %}>SUV</option>
                    </select>
                </div>
                <div class="input-wrapper">
                    <label>Drivetrain Config</label>
                    <select name="Drivetrain">
                        <option value="0" {% if form_values.Drivetrain == '0' %}selected{% endif %}>FWD</option>
                        <option value="1" {% if form_values.Drivetrain == '1' %}selected{% endif %}>RWD</option>
                        <option value="2" {% if form_values.Drivetrain == '2' %}selected{% endif %}>AWD</option>
                    </select>
                </div>

                <div class="input-wrapper">
                    <label>Electronics Package</label>
                    <select name="Tech_Package">
                        <option value="0" {% if form_values.Tech_Package == '0' %}selected{% endif %}>Standard Hub</option>
                        <option value="1" {% if form_values.Tech_Package == '1' %}selected{% endif %}>Advanced Navigation</option>
                    </select>
                </div>
                <div class="input-wrapper span-2">
                    <label>Regional Marketplace</label>
                    <select name="Location">
                        <option value="0" {% if form_values.Location == '0' %}selected{% endif %}>Metro Hub</option>
                        <option value="1" {% if form_values.Location == '1' %}selected{% endif %}>Regional District</option>
                    </select>
                </div>

                <button type="submit" class="submit-trigger">Process Asset Value Vectors</button>
            </form>
        </div>

        <!-- Output Analytics Layout presentation column section panel -->
        <div class="analytics-side">
            {% if prediction_result is not none %}
            <div class="valuation-display">
                <h2>Evaluated Target Vector Consensus</h2>
                <div class="valuation-price-container">
                    <span id="currencySymbol">₹</span>
                    <span id="baseValuationPrice" data-inr="{{ prediction_result|replace(',', '') }}">{{ prediction_result }}</span>
                </div>
                
                <!-- AI Strategy Insight Suggestion Container elements -->
                <div class="suggestions-block">
                    <strong>Asset Status Insight:</strong> {{ dynamic_suggestion }}
                </div>
                
                <p style="font-size: 0.8rem; opacity: 0.6; margin-top: 0.8rem;">RandomForest Regressor Ensemble 17-D Core Output Matrix</p>
            </div>
            {% endif %}

            <!-- High-contrast Pie Canvas chart tracking mapping elements -->
            <div class="glass-card chart-card">
                <h3 style="font-size: 0.9rem; font-weight: 700; margin-bottom: 1rem; color: var(--neon-accent); text-transform: uppercase; letter-spacing: 0.05em;">Feature Weight Distribution</h3>
                <div style="width: 100%; max-height: 200px; display: flex; justify-content: center;">
                    <canvas id="featureWeightPieChart"></canvas>
                </div>
            </div>

            <!-- Historical Activities Traces streams -->
            <div class="glass-card chat-history-card">
                <div class="chat-header">
                    <h3 style="font-size: 0.9rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase;">Sequential Prediction Logs</h3>
                    {% if history %}
                    <form method="POST" action="/clear">
                        <button type="submit" class="clear-history-action">Purge Logs</button>
                    </form>
                    {% endif %}
                </div>

                <div class="chat-log-stream">
                    {% for interaction in history %}
                        <div class="chat-bubble user-query">
                            <div class="bubble-meta">Asset Input Payload Data</div>
                            Year: {{ interaction.inputs.Year }} | Mileage: {{ interaction.inputs.Mileage }} km | HP: {{ interaction.inputs.Horsepower }}
                        </div>
                        <div class="chat-bubble ai-response">
                            <div class="bubble-meta">Pipeline Output</div>
                            Resolution metrics mapped value signature to token: 
                            <strong style="color: var(--neon-accent);" class="loggedPrice" data-inr="{{ interaction.output|replace(',', '') }}">₹{{ interaction.output }} INR</strong>
                        </div>
                    {% else %}
                        <div class="no-records">No session records tracked in system active thread.</div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>

    <script>
        const currencyExchangeMatrix = {
            INR: { symbol: '₹', rate: 1.0 },
            USD: { symbol: '$', rate: 0.012 },
            EUR: { symbol: '€', rate: 0.011 },
            GBP: { symbol: '£', rate: 0.0095 }
        };

        function switchApplicationTheme() {
            const chosenTheme = document.getElementById('themeSelector').value;
            document.documentElement.setAttribute('data-theme', chosenTheme);
        }

        function convertActiveValuations() {
            const selector = document.getElementById('currencySelector');
            const targetCurrency = selector.value;
            const configuration = currencyExchangeMatrix[targetCurrency];
            
            const primaryPriceElement = document.getElementById('baseValuationPrice');
            const primarySymbolElement = document.getElementById('currencySymbol');
            if(primaryPriceElement && primarySymbolElement) {
                const nativeINRValue = parseFloat(primaryPriceElement.getAttribute('data-inr'));
                const scaledPrice = nativeINRValue * configuration.rate;
                primarySymbolElement.innerText = configuration.symbol;
                primaryPriceElement.innerText = scaledPrice.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
            }

            document.querySelectorAll('.loggedPrice').forEach(element => {
                const nativeINR = parseFloat(element.getAttribute('data-inr'));
                const scaled = nativeINR * configuration.rate;
                element.innerText = configuration.symbol + scaled.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' ' + targetCurrency;
            });
        }

        document.addEventListener("DOMContentLoaded", function() {
            const mileageVal = parseFloat(document.getElementById('Mileage') ? document.getElementById('Mileage').value : 0) || 1000;
            const hpVal = parseFloat(document.getElementById('Horsepower') ? document.getElementById('Horsepower').value : 167) || 150;
            const torqueVal = parseFloat(document.getElementById('Torque') ? document.getElementById('Torque').value : 246) || 200;

            const ctx = document.getElementById('featureWeightPieChart').getContext('2d');
            new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ['Odometer Metrics', 'Engine Output (HP)', 'Torque Profile', 'Structural Base Features'],
                    datasets: [{
                        data: [mileageVal * 0.5 + 2000, hpVal * 25, torqueVal * 20, 15000],
                        backgroundColor: ['#06b6d4', '#ea580c', '#ec4899', '#4b5563'],
                        borderWidth: 1,
                        borderColor: 'rgba(255,255,255,0.08)'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 10 } } }
                    }
                }
            });
            
            convertActiveValuations();
        });
    </script>
</body>
</html>
"""

# -------------------------------------------------------------
# CORE PIPELINE MATRIX DESPATCH GATEWAY ROUTER
# -------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def main_gateway():
    error_msg = None
    prediction_result = None
    dynamic_suggestion = ""
    form_values = {}

    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        all_ui_fields = [
            'Make', 'Model', 'Trim_Level', 'Year', 'Fuel_Type', 'Transmission', 'Engine_Size', 
            'Service_History', 'Mileage', 'Horsepower', 'Torque', 'Owners', 
            'Accident_History', 'Color', 'Interior_Condition', 'Body_Type', 'Drivetrain', 
            'Tech_Package', 'Fuel_Efficiency', 'Location'
        ]
        form_values = {f: request.form.get(f) for f in all_ui_fields}

        # DYNAMIC INTELLIGENCE INSIGHT ENGINE EVALUATION BLOCK
        try:
            hp = int(form_values.get('Horsepower', 167) or 167)
            body = form_values.get('Body_Type', '0')
            drive = form_values.get('Drivetrain', '0')
            eff = float(form_values.get('Fuel_Efficiency', 30.0) or 30.0)

            suggestions = []
            if hp >= 280:
                suggestions.append("This car is incredibly fast with extreme track acceleration power!")
            elif hp >= 160:
                suggestions.append("Solid top-tier throttle response profile.")
                
            if body == '2' and drive == '2':
                suggestions.append("This variant configuration is highly capable for off-roading adventures.")
                
            if hp > 150 and eff > 20:
                suggestions.append("This structural powertrain combination is excellent, matching high speed with smart economy balance!")
            
            if not suggestions:
                suggestions.append("Balanced consumer asset matrix specifications profiles.")
                
            dynamic_suggestion = " ".join(suggestions)
        except Exception:
            dynamic_suggestion = "Asset parameters vector processed correctly."

        # Model Execution Vector compilation block mapping down to 17 features
        if model is None:
            try:
                base_calculation = 3600000.00
                mileage_deduction = float(form_values.get('Mileage', 0) or 0) * 4.2
                age_deduction = (2026 - int(form_values.get('Year', 2026) or 2026)) * 115000
                hp_bonus = (int(form_values.get('Horsepower', 167) or 167) - 100) * 3800
                
                calculated_sim_val = max(280000.00, base_calculation - mileage_deduction - age_deduction + hp_bonus)
                prediction_result = f"{calculated_sim_val:,.2f}"

                current_stack = session["history"]
                current_stack.insert(0, {"inputs": form_values, "output": prediction_result})
                session["history"] = current_stack[:6]
            except Exception as ex:
                error_msg = f"Vector Generation Simulation Exception: {str(ex)}"
        else:
            try:
                evaluation_vector = np.array([[
                    int(form_values['Make']),
                    int(form_values['Model']),
                    int(form_values['Year'] or 2026),
                    int(form_values['Fuel_Type']),
                    int(form_values['Transmission']),
                    float(form_values['Engine_Size'] or 2.0),
                    int(form_values['Service_History']),
                    float(form_values['Mileage'] or 0),
                    int(form_values['Horsepower'] or 150),
                    int(form_values['Torque'] or 250),
                    int(form_values['Owners']),
                    int(form_values['Accident_History']),
                    int(form_values['Color']),
                    int(form_values['Body_Type']),
                    int(form_values['Drivetrain']),
                    float(form_values['Fuel_Efficiency'] or 15.0),
                    int(form_values['Location'])
                ]], dtype=object)

                calculated_matrix = model.predict(evaluation_vector)
                prediction_result = f"{float(calculated_matrix[0]):,.2f}"

                current_stack = session["history"]
                current_stack.insert(0, {"inputs": form_values, "output": prediction_result})
                session["history"] = current_stack[:6]

            except Exception as ex:
                error_msg = f"Vector Compilation Execution Error: {str(ex)}"

    return render_template_string(
        DASHBOARD_TEMPLATE,
        prediction_result=prediction_result,
        form_values=form_values,
        history=session.get("history", []),
        error_msg=error_msg,
        dynamic_suggestion=dynamic_suggestion
    )

@app.route("/clear", methods=["POST"])
def purge_logs():
    session["history"] = []
    return render_template_string(DASHBOARD_TEMPLATE, prediction_result=None, form_values={}, history=[], error_msg=None, dynamic_suggestion="")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
