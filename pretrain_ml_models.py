import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def pretrain():
    print("=== PRE-TRAINING ML MODELS LOCALLY ===")
    os.makedirs('models', exist_ok=True)
    
    # Load dataset
    df = pd.read_csv("data/crop_recommendation.csv")
    print(f"Loaded dataset: {df.shape}")
    
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    # Preprocessors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save preprocessors
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("[OK] Saved scaler.pkl and label_encoder.pkl")
    
    # 1. Random Forest (with OOB evaluation for accuracy)
    print("Training Random Forest Classifier (150 estimators)...")
    rf_model = RandomForestClassifier(n_estimators=150, random_state=42, oob_score=True)
    rf_model.fit(X_scaled, y)
    rf_accuracy = rf_model.oob_score_
    print(f"[OK] Random Forest OOB Accuracy: {rf_accuracy:.2%}")
    
    # Save Random Forest Model
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("[OK] Saved random_forest_model.pkl")
    
    # 2. XGBoost (with Label Encoding)
    print("Training XGBoost Classifier (150 estimators)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    xgb_model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_accuracy = xgb_model.score(X_test, y_test)
    print(f"[OK] XGBoost Validation Accuracy: {xgb_accuracy:.2%}")
    
    # Retrain on full dataset
    xgb_model_full = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model_full.fit(X_scaled, y_encoded)
    
    # Save XGBoost Model
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model_full, f)
    print("[OK] Saved xgboost_model.pkl")
    
    # Save accuracies to JSON
    accuracies = {
        "Random Forest": float(rf_accuracy),
        "XGBoost": float(xgb_accuracy)
    }
    with open('models/model_accuracies.json', 'w') as f:
        json.dump(accuracies, f, indent=4)
    print("[OK] Saved model_accuracies.json")
    
    print("=== PRE-TRAINING SUCCESSFULLY COMPLETED ===")

if __name__ == "__main__":
    pretrain()
