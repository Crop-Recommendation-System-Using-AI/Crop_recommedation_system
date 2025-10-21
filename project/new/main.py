from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests
import json

app = Flask(__name__)

# Load the trained models
def load_models():
    model = pickle.load(open('crop_recommendation_model.pkl', 'rb'))
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    production_summary = pd.read_csv('production_summary.csv')
    return model, label_encoder, scaler, production_summary

# Load models at startup
model, label_encoder, scaler, production_summary = load_models()

# Function to get AI recommendations from Ollama
def get_ai_recommendations(crop_name):
    # Prepare the prompt for Ollama
    prompt = f"Give detailed crop recommendations for {crop_name} and provide maintenance ideas for farmers."
    
    # Make a request to Ollama API
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama2',  # You can change this to any model available in your Ollama instance
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
        return f"Error connecting to Ollama server: {str(e)}"

# Function to get production data for a crop
def get_production_data(crop_name):
    crop_data = production_summary[production_summary['Crop'].str.lower() == crop_name.lower()]
    
    if crop_data.empty:
        return "No production data available for this crop."
    
    # Group by state to get average yield
    state_data = crop_data.groupby('State_Name').agg({
        'Area': 'sum',
        'Production': 'sum',
        'Yield_per_Area': 'mean'
    }).reset_index()
    
    # Sort by yield
    state_data = state_data.sort_values('Yield_per_Area', ascending=False)
    
    # Convert to dictionary for JSON response
    result = {
        'states': state_data['State_Name'].tolist(),
        'yields': state_data['Yield_per_Area'].tolist(),
        'total_area': state_data['Area'].sum(),
        'total_production': state_data['Production'].sum()
    }
    
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    N = float(request.form['nitrogen'])
    P = float(request.form['phosphorous'])
    K = float(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    
    # Create input array
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    predicted_crop = label_encoder.inverse_transform(prediction)[0]
    
    # Get AI recommendations
    ai_recommendations = get_ai_recommendations(predicted_crop)
    
    # Get production data
    production_data = get_production_data(predicted_crop)
    
    # Return the results
    return render_template('result.html', 
                          predicted_crop=predicted_crop,
                          ai_recommendations=ai_recommendations,
                          production_data=production_data)

@app.route('/api/predict', methods=['POST'])
def api_predict():
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
    
    # Create input array
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    predicted_crop = label_encoder.inverse_transform(prediction)[0]
    
    # Get AI recommendations
    ai_recommendations = get_ai_recommendations(predicted_crop)
    
    # Get production data
    production_data = get_production_data(predicted_crop)
    
    # Return the results as JSON
    return jsonify({
        'predicted_crop': predicted_crop,
        'ai_recommendations': ai_recommendations,
        'production_data': production_data
    })

if __name__ == '__main__':
    app.run(debug=True)