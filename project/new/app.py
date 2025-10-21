from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from markupsafe import Markup  # Import Markup from the correct location
import pickle
import numpy as np
import pandas as pd
import requests
import json
import os
import re

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key
app.jinja_env.globals.update(min=min)
# Function to format text with bold words
def format_recommendation_text(text):
    # Replace **text** with <strong>text</strong>
    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Replace *text* with <em>text</em>
    formatted_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted_text)
    
    # Convert line breaks to <br>
    formatted_text = formatted_text.replace('\n', '<br>')
    
    return formatted_text

# Check if models exist, if not create dummy models
def check_and_create_models():
    if not os.path.exists('crop_recommendation_model.pkl'):
        print("Model files not found. Creating dummy models for demonstration.")
        
        # Create a simple dummy model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        
        # Create dummy data
        np.random.seed(42)
        n_samples = 1000
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = pd.DataFrame(np.random.rand(n_samples, 7) * 100, columns=feature_names)
        y = np.random.randint(0, 10, n_samples)
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Create label encoder with some crop names
        crop_names = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'potato', 'onion', 'tomato', 'chili', 'banana']
        le = LabelEncoder()
        le.fit(crop_names)
        
        # Create scaler
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Save the models
        pickle.dump(model, open('crop_recommendation_model.pkl', 'wb'))
        pickle.dump(le, open('label_encoder.pkl', 'wb'))
        pickle.dump(scaler, open('scaler.pkl', 'wb'))
        
        # Create dummy production data with more realistic state names
        states = ['California', 'Texas', 'Iowa', 'Nebraska', 'Kansas', 'Minnesota', 'Illinois', 'Indiana', 'Ohio', 'Missouri']
        production_data = []
        
        for crop in crop_names:
            for state in states:
                area = np.random.randint(100, 10000)
                production = np.random.randint(1000, 100000)
                yield_per_area = production / area if area > 0 else 0
                
                production_data.append({
                    'State_Name': state,
                    'District_Name': f'District in {state}',
                    'Crop': crop,
                    'Area': area,
                    'Production': production,
                    'Yield_per_Area': yield_per_area
                })
        
        production_df = pd.DataFrame(production_data)
        production_df.to_csv('production_summary.csv', index=False)
        
        print("Dummy models created successfully!")

# Load the trained models
def load_models():
    try:
        model = pickle.load(open('crop_recommendation_model.pkl', 'rb'))
        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        production_summary = pd.read_csv('production_summary.csv')
        return model, label_encoder, scaler, production_summary
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None

# Function to get AI recommendations from Ollama
def get_ai_recommendations(crop_name):
    # Prepare the prompt for Ollama
    prompt = f"Give detailed crop recommendations for {crop_name} and provide maintenance ideas for farmers in india."
    
    # Make a request to Ollama API
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'Jayasimma/Puzhavan',  # You can change this to any model available in your Ollama instance
                'prompt': prompt,
                'stream': False
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'No response from AI model.')
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        # If Ollama is not available, return a generic recommendation
        if crop_name.lower() == 'banana':
            return """**I. Banana Variety Recommendations (Based on Region & Goals)**

The best banana variety depends heavily on your climate, soil, market demand, and desired characteristics (e.g., sweetness, disease resistance). Here's a breakdown:

* **For Tropical Climates (High Heat & Humidity - e.g., Southeast Asia, India, parts of Africa):**
  * **Cavendish (Grand Nain & Williams):** The most widely grown globally. Grand Nain is sweeter, Williams is more robust. Highly susceptible to Panama Disease (Tropical Race 4 - TR4).
  * **Manzano:** Small, apple-flavored bananas. Good for fresh markets. Relatively susceptible to diseases.
  * **Pisang Awak:** Very hardy, disease-resistant, and produces fruit year-round. The fruit is starchy and needs cooking. Popular in Indonesia and Malaysia.

* **For Subtropical Climates (Moderate Temperatures - e.g., Southern California, Florida, parts of Australia):**
  * **Dwarf Cavendish:** More heat tolerant than standard Cavendish.
  * **Goldfinger:** Excellent heat and cold tolerance. Good for areas with fluctuating temperatures.
  * **Ice Cream Banana (Rajapuri):** Becoming increasingly popular for its creamy texture and flavor.

* **For Cooler Climates (e.g., parts of New Zealand, Southern Europe):**
  * **Silk Hope:** Highly productive and tolerant of cooler temperatures.
  * **Portuguese Cavendish:** Good for cooler, humid conditions.

**II. Soil & Nutrient Management**

* **Soil Type:** Bananas thrive in well-drained, fertile soils. Loamy soils are ideal. Clay soils need significant amendment with organic matter. Sandy soils require frequent irrigation.
* **pH:** 6.0 - 7.5 is optimal.
* **Nutrient Requirements:** Bananas are heavy feeders.
  * **Nitrogen (N):** Crucial for vegetative growth. Apply in split doses – a larger dose at planting and smaller, regular applications throughout the growing season.
  * **Phosphorus (P):** Important for root development and flowering.
  * **Potassium (K):** Essential for fruit development and quality.
  * **Micronutrients:** Ensure adequate levels of zinc, boron, manganese, and iron. Boron is particularly critical for fruit set.
* **Organic Matter:** Incorporate compost, manure, or green manure crops to improve soil structure, fertility, and water retention.

**III. Irrigation & Water Management**

* **Water Needs:** Bananas require consistent moisture, especially during flowering and fruit development.
* **Irrigation Methods:**
  * **Drip Irrigation:** Most efficient, delivering water directly to the root zone, minimizing water loss and weed growth.
  * **Sprinkler Irrigation:** Less efficient than drip, but can be used in larger fields.
* **Water Quality:** Avoid using water with high salinity.

**IV. Pest & Disease Management**

* **Panama Disease (TR4):** The biggest threat. Strict quarantine measures are vital. Resistant varieties are key. Monitor regularly and implement early detection programs.
* **Weevils:** Banana weevils are a major pest. Use traps, biological control (e.g., nematodes), and resistant varieties.
* **Sigatoka:** A fungal disease causing leaf spots. Regular fungicide applications may be necessary.
* **Other Pests:** Thrips, aphids, and nematodes can also cause problems.
* **Integrated Pest Management (IPM):** A crucial approach combining biological controls, cultural practices, and targeted pesticide applications when necessary.

**V. Pruning & Training**

* **Initial Training:** Use stakes or props to support young plants.
* **Pseudostem Pruning:** Regularly remove lower pseudostems (the leafy stems) to improve light penetration and air circulation. This encourages new growth at the top.
* **Bunch Removal:** Remove old, unproductive bunches to stimulate the production of new ones.

**VI. Harvesting & Post-Harvest Handling**

* **Harvesting:** Bananas are typically harvested when they are green and firm. The time to harvest depends on the variety and desired market.
* **Post-Harvest Handling:** Proper handling is essential to maintain quality. Bananas should be cooled quickly after harvest to slow down ripening.

**VII. Maintenance Ideas & Best Practices**

* **Regular Soil Testing:** At least annually to monitor nutrient levels and pH.
* **Record Keeping:** Maintain detailed records of planting dates, fertilizer applications, pest and disease outbreaks, and yields.
* **Farmer Training:** Invest in training programs to educate farmers on best practices.
* **Community Collaboration:** Establish farmer groups for sharing knowledge and resources.
* **Crop Rotation:** Consider rotating banana crops with other crops to improve soil health.

**Disclaimer:** This information is for general guidance only. Specific recommendations will vary depending on your location, soil type, and market demands. Consult with a local agricultural extension officer or agricultural specialist for tailored advice."""
        else:
            return f"For growing {crop_name}, ensure proper soil preparation, adequate irrigation, regular monitoring for pests and diseases, and timely application of fertilizers. Consult with local agricultural experts for region-specific advice."

# Function to get production data for a crop
def get_production_data(crop_name, production_summary):
    if production_summary is None:
        return {"error": "No production data available."}
    
    crop_data = production_summary[production_summary['Crop'].str.lower() == crop_name.lower()]
    
    if crop_data.empty:
        return {"error": "No production data available for this crop."}
    
    # Group by state to get average yield
    state_data = crop_data.groupby('State_Name').agg({
        'Area': 'sum',
        'Production': 'sum',
        'Yield_per_Area': 'mean'
    }).reset_index()
    
    # Sort by yield
    state_data = state_data.sort_values('Yield_per_Area', ascending=False)
    
    # Get top 5 states
    top_states = state_data.head(5)
    top_states_list = list(zip(top_states['State_Name'], top_states['Yield_per_Area']))
    
    # Convert to dictionary for JSON response
    result = {
        'states': state_data['State_Name'].tolist(),
        'yields': state_data['Yield_per_Area'].tolist(),
        'total_area': state_data['Area'].sum(),
        'total_production': state_data['Production'].sum(),
        'top_states': top_states_list
    }
    
    return result

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/register', methods=['POST'])
def register():
    # Get form data
    firstName = request.form['firstName']
    lastName = request.form['lastName']
    email = request.form['email']
    password = request.form['password']
    farmLocation = request.form['farmLocation']
    farmSize = request.form['farmSize']
    
    # Here you would typically save the user data to a database
    # For now, we'll just show a success message
    
    flash('Account created successfully! You can now log in.', 'success')
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Create input DataFrame with column names
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]
        
        # Get AI recommendations
        ai_recommendations = get_ai_recommendations(predicted_crop)
        
        # Format the recommendations with bold text
        formatted_recommendations = format_recommendation_text(ai_recommendations)
        
        # Get production data
        production_data = get_production_data(predicted_crop, production_summary)
        
        # Return the results
        return render_template('result.html', 
                              predicted_crop=predicted_crop,
                              ai_recommendations=Markup(formatted_recommendations),
                              production_data=production_data)
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Extract values
        N = float(data['nitrogen'])
        P = float(data['phosphorous'])
        K = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
        # Create input DataFrame with column names
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]
        
        # Get AI recommendations
        ai_recommendations = get_ai_recommendations(predicted_crop)
        
        # Format the recommendations with bold text
        formatted_recommendations = format_recommendation_text(ai_recommendations)
        
        # Get production data
        production_data = get_production_data(predicted_crop, production_summary)
        
        # Return the results as JSON
        return jsonify({
            'predicted_crop': predicted_crop,
            'ai_recommendations': formatted_recommendations,
            'production_data': production_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Check and create models if they don't exist
    check_and_create_models()
    
    # Load models
    model, label_encoder, scaler, production_summary = load_models()
    
    if model is None:
        print("Failed to load models. Exiting.")
        exit(1)
    
    app.run(debug=True)