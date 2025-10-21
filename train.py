import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
def load_datasets():
    # Load production dataset
    production_data = pd.read_csv('production_data.csv')
    
    # Load recommendation dataset
    recommendation_data = pd.read_csv('recommendation_data.csv')
    
    return production_data, recommendation_data

# Preprocess the recommendation dataset
def preprocess_recommendation_data(df):
    # Check for missing values
    print("Missing values in recommendation dataset:")
    print(df.isnull().sum())
    
    # Encode the label column (crop names)
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    
    # Separate features and target
    X = df.drop(['label', 'label_encoded'], axis=1)
    y = df['label_encoded']
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, le, scaler

# Train the recommendation model
def train_recommendation_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    return model

# Preprocess the production dataset
def preprocess_production_data(df):
    # Check for missing values
    print("Missing values in production dataset:")
    print(df.isnull().sum())
    
    # Fill missing values if any
    df = df.fillna(0)
    
    # Group by state, district, and crop to get average production
    production_summary = df.groupby(['State_Name', 'District_Name', 'Crop']).agg({
        'Area': 'sum',
        'Production': 'sum'
    }).reset_index()
    
    # Calculate yield per unit area
    production_summary['Yield_per_Area'] = production_summary['Production'] / production_summary['Area']
    production_summary = production_summary.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return production_summary

# Save the models and preprocessors
def save_models(model, label_encoder, scaler):
    pickle.dump(model, open('crop_recommendation_model.pkl', 'wb'))
    pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    print("Models saved successfully!")

# Main function to run the training process
def main():
    # Load datasets
    production_data, recommendation_data = load_datasets()
    
    # Preprocess recommendation dataset
    X, y, label_encoder, scaler = preprocess_recommendation_data(recommendation_data)
    
    # Train the recommendation model
    model = train_recommendation_model(X, y)
    
    # Save the models
    save_models(model, label_encoder, scaler)
    
    # Preprocess production dataset
    production_summary = preprocess_production_data(production_data)
    
    # Save the production summary
    production_summary.to_csv('production_summary.csv', index=False)
    print("Production summary saved successfully!")

if __name__ == "__main__":
    main()