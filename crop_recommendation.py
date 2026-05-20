#!/usr/bin/env python3
"""
Crop Advisory System - Main Implementation
Data-Driven Crop Recommendation using Random Forest and Regression
Supports 36 crops and a large dataset of 18,000 samples (500 samples per crop).
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def create_crop_dataset():
    """Create a comprehensive crop recommendation dataset with 36 crops and 500 samples each (18000 total)"""
    np.random.seed(42)
    
    crop_conditions = {
        # 1-10: Original/First batch
        'rice': {'N': (80, 120), 'P': (35, 60), 'K': (35, 45), 'temp': (20, 35), 'humidity': (80, 95), 'ph': (5.5, 7.0), 'rainfall': (150, 300)},
        'maize': {'N': (80, 120), 'P': (40, 70), 'K': (40, 70), 'temp': (18, 27), 'humidity': (60, 70), 'ph': (5.8, 7.5), 'rainfall': (50, 100)},
        'chickpea': {'N': (40, 70), 'P': (60, 80), 'K': (80, 120), 'temp': (20, 25), 'humidity': (10, 40), 'ph': (6.0, 7.5), 'rainfall': (30, 100)},
        'kidneybeans': {'N': (20, 40), 'P': (60, 80), 'K': (50, 70), 'temp': (15, 25), 'humidity': (18, 65), 'ph': (5.5, 7.0), 'rainfall': (60, 120)},
        'pigeonpeas': {'N': (20, 40), 'P': (60, 80), 'K': (60, 80), 'temp': (18, 29), 'humidity': (30, 65), 'ph': (4.5, 8.2), 'rainfall': (60, 180)},
        'mothbeans': {'N': (15, 30), 'P': (40, 60), 'K': (20, 40), 'temp': (25, 35), 'humidity': (40, 65), 'ph': (5.5, 8.0), 'rainfall': (30, 70)},
        'mungbean': {'N': (15, 28), 'P': (35, 50), 'K': (15, 28), 'temp': (25, 30), 'humidity': (40, 60), 'ph': (6.0, 7.0), 'rainfall': (40, 65)},
        'blackgram': {'N': (15, 30), 'P': (55, 70), 'K': (30, 45), 'temp': (31, 35), 'humidity': (61, 75), 'ph': (6.5, 7.5), 'rainfall': (70, 90)},
        'lentil': {'N': (15, 30), 'P': (50, 70), 'K': (30, 50), 'temp': (15, 25), 'humidity': (20, 50), 'ph': (5.5, 7.0), 'rainfall': (30, 80)},
        'pomegranate': {'N': (30, 50), 'P': (10, 30), 'K': (30, 50), 'temp': (20, 35), 'humidity': (30, 60), 'ph': (5.5, 7.5), 'rainfall': (50, 100)},
        # 11-20: Second batch
        'banana': {'N': (100, 120), 'P': (75, 90), 'K': (50, 60), 'temp': (26, 30), 'humidity': (75, 85), 'ph': (5.5, 7.0), 'rainfall': (100, 180)},
        'mango': {'N': (20, 40), 'P': (20, 40), 'K': (30, 50), 'temp': (27, 35), 'humidity': (50, 70), 'ph': (5.5, 7.0), 'rainfall': (80, 150)},
        'grapes': {'N': (20, 40), 'P': (120, 140), 'K': (200, 250), 'temp': (8, 15), 'humidity': (80, 90), 'ph': (5.5, 7.0), 'rainfall': (50, 90)},
        'watermelon': {'N': (100, 120), 'P': (80, 120), 'K': (40, 50), 'temp': (22, 26), 'humidity': (80, 90), 'ph': (6.0, 7.0), 'rainfall': (65, 80)},
        'muskmelon': {'N': (100, 120), 'P': (80, 120), 'K': (40, 50), 'temp': (28, 33), 'humidity': (80, 90), 'ph': (6.0, 7.0), 'rainfall': (30, 45)},
        'apple': {'N': (20, 40), 'P': (120, 140), 'K': (200, 250), 'temp': (16, 25), 'humidity': (90, 95), 'ph': (5.5, 6.5), 'rainfall': (120, 250)},
        'orange': {'N': (20, 40), 'P': (10, 30), 'K': (10, 30), 'temp': (15, 35), 'humidity': (90, 95), 'ph': (6.0, 8.0), 'rainfall': (100, 200)},
        'papaya': {'N': (40, 60), 'P': (45, 65), 'K': (45, 65), 'temp': (23, 35), 'humidity': (80, 90), 'ph': (5.5, 7.0), 'rainfall': (150, 250)},
        'coconut': {'N': (20, 40), 'P': (10, 25), 'K': (30, 45), 'temp': (25, 30), 'humidity': (55, 65), 'ph': (5.0, 6.5), 'rainfall': (150, 300)},
        'cotton': {'N': (120, 160), 'P': (40, 60), 'K': (40, 60), 'temp': (21, 30), 'humidity': (50, 80), 'ph': (5.8, 8.0), 'rainfall': (50, 100)},
        # 21-22: Alluvial/Highland
        'jute': {'N': (70, 90), 'P': (35, 50), 'K': (35, 50), 'temp': (24, 35), 'humidity': (70, 90), 'ph': (6.0, 7.8), 'rainfall': (150, 250)},
        'coffee': {'N': (90, 110), 'P': (15, 30), 'K': (25, 40), 'temp': (23, 28), 'humidity': (50, 60), 'ph': (6.0, 7.0), 'rainfall': (130, 200)},
        # 23-36: New Crops Added for Area Scalability
        'wheat': {'N': (60, 80), 'P': (45, 60), 'K': (35, 50), 'temp': (15, 22), 'humidity': (55, 70), 'ph': (6.2, 7.5), 'rainfall': (60, 90)},
        'barley': {'N': (40, 55), 'P': (30, 42), 'K': (30, 42), 'temp': (10, 18), 'humidity': (40, 55), 'ph': (6.0, 7.0), 'rainfall': (35, 55)},
        'sugarcane': {'N': (120, 180), 'P': (50, 80), 'K': (60, 100), 'temp': (20, 35), 'humidity': (60, 85), 'ph': (6.0, 7.8), 'rainfall': (120, 250)},
        'ragi': {'N': (40, 60), 'P': (30, 40), 'K': (30, 40), 'temp': (20, 30), 'humidity': (50, 70), 'ph': (5.0, 8.2), 'rainfall': (45, 80)},
        'bajra': {'N': (35, 50), 'P': (25, 40), 'K': (25, 40), 'temp': (31, 37), 'humidity': (40, 60), 'ph': (6.5, 8.5), 'rainfall': (25, 45)},
        'jowar': {'N': (55, 75), 'P': (30, 45), 'K': (30, 45), 'temp': (26, 30), 'humidity': (40, 55), 'ph': (6.5, 8.0), 'rainfall': (50, 80)},
        'mustard': {'N': (70, 90), 'P': (35, 50), 'K': (30, 42), 'temp': (18, 25), 'humidity': (50, 65), 'ph': (6.0, 7.2), 'rainfall': (30, 50)},
        'soybean': {'N': (20, 35), 'P': (60, 80), 'K': (30, 50), 'temp': (18, 30), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (60, 120)},
        'groundnut': {'N': (20, 40), 'P': (40, 60), 'K': (20, 40), 'temp': (22, 30), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (50, 100)},
        'sunflower': {'N': (50, 80), 'P': (40, 60), 'K': (40, 60), 'temp': (20, 28), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (50, 100)},
        'potato': {'N': (80, 120), 'P': (60, 90), 'K': (100, 140), 'temp': (15, 22), 'humidity': (70, 95), 'ph': (5.2, 6.5), 'rainfall': (60, 120)},
        'tomato': {'N': (70, 100), 'P': (50, 80), 'K': (60, 90), 'temp': (18, 28), 'humidity': (60, 85), 'ph': (5.8, 7.0), 'rainfall': (60, 120)},
        'onion': {'N': (60, 90), 'P': (30, 50), 'K': (40, 70), 'temp': (15, 25), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (50, 100)},
        'tea': {'N': (100, 140), 'P': (20, 40), 'K': (30, 50), 'temp': (18, 30), 'humidity': (70, 90), 'ph': (4.5, 5.5), 'rainfall': (150, 300)}
    }
    
    data = []
    crops = list(crop_conditions.keys())
    
    # Generate 500 samples per crop (18000 total)
    for crop in crops:
        conditions = crop_conditions[crop]
        for _ in range(500):
            N = np.random.uniform(conditions['N'][0], conditions['N'][1]) + np.random.normal(0, 0.5)
            P = np.random.uniform(conditions['P'][0], conditions['P'][1]) + np.random.normal(0, 0.5)
            K = np.random.uniform(conditions['K'][0], conditions['K'][1]) + np.random.normal(0, 0.5)
            temperature = np.random.uniform(conditions['temp'][0], conditions['temp'][1]) + np.random.normal(0, 0.1)
            humidity = np.random.uniform(conditions['humidity'][0], conditions['humidity'][1]) + np.random.normal(0, 0.5)
            ph = np.random.uniform(conditions['ph'][0], conditions['ph'][1]) + np.random.normal(0, 0.02)
            rainfall = np.random.uniform(conditions['rainfall'][0], conditions['rainfall'][1]) + np.random.normal(0, 1.0)
            
            N, P, K = max(0, N), max(0, P), max(0, K)
            temperature = max(0, temperature)
            humidity = max(0, min(100, humidity))
            ph = max(0, min(14, ph))
            rainfall = max(0, rainfall)
            
            data.append([N, P, K, temperature, humidity, ph, rainfall, crop])
            
    df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])
    return df

def analyze_dataset(df):
    """Analyze the final dataset shape and profiles"""
    print("\n=== DATASET ANALYSIS PROFILE ===")
    print(f"Dataset Dimensions: {df.shape}")
    print(f"Total Unique Crops Supported: {df['label'].nunique()}")
    
    crop_averages = df.groupby('label').mean()
    
    # Save averages for application reference
    os.makedirs('data', exist_ok=True)
    crop_averages.to_csv('data/crop_averages.csv')
    print("\n[OK] Saved crop averages matrix to 'data/crop_averages.csv'")
    
    return df

def train_models(df):
    """Train RF and LR algorithms"""
    print("\n=== TRAINING MACHINE LEARNING MODELS ===")
    
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    lr_model = LogisticRegression(max_iter=3000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    
    print(f"Random Forest Model Accuracy: {rf_acc:.2%}")
    print(f"Logistic Regression Model Accuracy: {lr_acc:.2%}")
    
    return {
        'rf_model': rf_model,
        'scaler': scaler,
        'rf_accuracy': rf_acc,
        'lr_accuracy': lr_acc
    }

def main():
    print("==================================================")
    print("CROP ADVISORY PIPELINE: EXPANSION & ML RE-TRAINING (18,000 SAMPLES)")
    print("==================================================")
    
    print("Generating comprehensive dataset of 18,000 samples with 36 crops...")
    df = create_crop_dataset()
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/crop_recommendation.csv', index=False)
    print("[OK] Saved comprehensive dataset to 'data/crop_recommendation.csv'")
    
    df = analyze_dataset(df)
    models_data = train_models(df)
    print("\n[OK] Model pipeline execution completed successfully!")

if __name__ == "__main__":
    main()