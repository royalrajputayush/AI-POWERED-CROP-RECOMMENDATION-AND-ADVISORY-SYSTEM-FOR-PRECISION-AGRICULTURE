#!/usr/bin/env python3
"""
Crop Advisory System Data Setup
Downloads and prepares the crop recommendation dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Create sample data if dataset not available
def create_sample_dataset():
    """Create a sample crop recommendation dataset"""
    np.random.seed(42)
    
    # Define crops
    crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 
             'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
             'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
             'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
    
    n_samples = 2200
    data = []
    
    for _ in range(n_samples):
        # Generate soil and environmental parameters
        N = np.random.normal(50, 20)  # Nitrogen
        P = np.random.normal(50, 20)  # Phosphorus
        K = np.random.normal(50, 20)  # Potassium
        temperature = np.random.normal(25, 8)  # Temperature in Celsius
        humidity = np.random.normal(70, 15)  # Humidity percentage
        ph = np.random.normal(6.5, 1)  # pH level
        rainfall = np.random.normal(150, 50)  # Rainfall in mm
        
        # Simple logic to assign crops based on conditions
        if temperature > 30 and humidity > 80:
            label = np.random.choice(['rice', 'cotton', 'jute'])
        elif temperature < 15 and rainfall < 100:
            label = np.random.choice(['apple', 'grapes'])
        elif ph < 6:
            label = np.random.choice(['rice', 'maize'])
        elif ph > 7:
            label = np.random.choice(['chickpea', 'kidneybeans'])
        else:
            label = np.random.choice(crops)
        
        data.append([N, P, K, temperature, humidity, ph, rainfall, label])
    
    df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])
    return df

def main():
    print("Setting up Crop Advisory System Dataset...")
    
    # Try to load existing dataset, otherwise create sample
    try:
        df = pd.read_csv('data/Crop_recommendation.csv')
        print("✓ Loaded existing dataset")
    except FileNotFoundError:
        print("Creating sample dataset...")
        df = create_sample_dataset()
        df.to_csv('data/crop_recommendation.csv', index=False)
        print("✓ Sample dataset created")
    
    # Basic dataset information
    print(f"\nDataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique Crops: {df['label'].nunique()}")
    print(f"Crop Types: {sorted(df['label'].unique())}")
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Basic statistics
    print("\nDataset Statistics:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    df = main()