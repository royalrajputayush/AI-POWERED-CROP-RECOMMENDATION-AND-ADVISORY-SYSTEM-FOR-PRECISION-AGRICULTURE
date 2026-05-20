#!/usr/bin/env python3
"""
Simple Crop Advisory System - Basic Implementation
Uses built-in Python libraries for demonstration
"""

import csv
import random
import math
from collections import defaultdict, Counter

def create_simple_dataset():
    """Create a simple crop recommendation dataset using basic Python"""
    random.seed(42)
    
    # Define crops with their optimal conditions
    crop_conditions = {
        'rice': {'N': 100, 'P': 50, 'K': 50, 'temp': 27, 'humidity': 85, 'ph': 6.2, 'rainfall': 225},
        'maize': {'N': 100, 'P': 60, 'K': 60, 'temp': 22, 'humidity': 65, 'ph': 6.9, 'rainfall': 75},
        'chickpea': {'N': 55, 'P': 70, 'K': 100, 'temp': 22, 'humidity': 25, 'ph': 6.8, 'rainfall': 65},
        'kidneybeans': {'N': 30, 'P': 70, 'K': 60, 'temp': 20, 'humidity': 41, 'ph': 6.2, 'rainfall': 90},
        'cotton': {'N': 140, 'P': 50, 'K': 50, 'temp': 25, 'humidity': 65, 'ph': 6.9, 'rainfall': 75},
        'banana': {'N': 110, 'P': 80, 'K': 55, 'temp': 28, 'humidity': 80, 'ph': 6.2, 'rainfall': 140},
        'watermelon': {'N': 110, 'P': 100, 'K': 45, 'temp': 25, 'humidity': 85, 'ph': 6.5, 'rainfall': 45},
        'grapes': {'N': 30, 'P': 130, 'K': 225, 'temp': 15, 'humidity': 85, 'ph': 6.2, 'rainfall': 87},
        'apple': {'N': 25, 'P': 130, 'K': 225, 'temp': 15, 'humidity': 85, 'ph': 6.2, 'rainfall': 87}
    }
    
    data = []
    header = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    
    # Generate samples for each crop
    for crop, conditions in crop_conditions.items():
        for _ in range(200):  # 200 samples per crop
            # Add some random variation to the optimal conditions
            sample = []
            for param in ['N', 'P', 'K', 'temp', 'humidity', 'ph', 'rainfall']:
                optimal = conditions[param]
                # Add random noise (±20% of optimal value)
                noise = random.uniform(-0.2, 0.2) * optimal
                value = max(0, optimal + noise)
                
                # Special handling for pH (0-14 range) and humidity (0-100 range)
                if param == 'ph':
                    value = max(0, min(14, value))
                elif param == 'humidity':
                    value = max(0, min(100, value))
                
                sample.append(round(value, 2))
            
            sample.append(crop)
            data.append(sample)
    
    # Shuffle the data
    random.shuffle(data)
    
    # Save to CSV
    with open('data/crop_recommendation.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)
    
    print(f"✓ Dataset created with {len(data)} samples and {len(crop_conditions)} crops")
    return data, header

def load_dataset():
    """Load the dataset from CSV file"""
    try:
        with open('data/crop_recommendation.csv', 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            data = []
            for row in reader:
                # Convert numeric values
                numeric_row = []
                for i, value in enumerate(row):
                    if i < len(row) - 1:  # All columns except label
                        numeric_row.append(float(value))
                    else:  # Label column
                        numeric_row.append(value)
                data.append(numeric_row)
        return data, header
    except FileNotFoundError:
        print("Dataset not found. Creating new dataset...")
        return create_simple_dataset()

def analyze_dataset(data, header):
    """Analyze the dataset using basic Python"""
    print("\n=== DATASET ANALYSIS ===")
    print(f"Total samples: {len(data)}")
    
    # Count crops
    crops = [row[-1] for row in data]
    crop_counts = Counter(crops)
    print(f"Number of unique crops: {len(crop_counts)}")
    print("Crop distribution:")
    for crop, count in crop_counts.items():
        print(f"  {crop}: {count} samples")
    
    # Basic statistics for numeric columns
    print("\nBasic Statistics:")
    for i, col_name in enumerate(header[:-1]):  # Exclude label column
        values = [row[i] for row in data]
        print(f"{col_name}:")
        print(f"  Min: {min(values):.2f}, Max: {max(values):.2f}")
        print(f"  Mean: {sum(values)/len(values):.2f}")

def simple_distance(sample1, sample2):
    """Calculate Euclidean distance between two samples"""
    distance = 0
    for i in range(len(sample1) - 1):  # Exclude label
        distance += (sample1[i] - sample2[i]) ** 2
    return math.sqrt(distance)

def simple_knn_classifier(train_data, test_sample, k=5):
    """Simple K-Nearest Neighbors classifier"""
    distances = []
    
    for train_sample in train_data:
        dist = simple_distance(train_sample, test_sample)
        distances.append((dist, train_sample[-1]))  # (distance, label)
    
    # Sort by distance and get k nearest neighbors
    distances.sort()
    nearest_neighbors = distances[:k]
    
    # Count votes
    votes = Counter([neighbor[1] for neighbor in nearest_neighbors])
    prediction = votes.most_common(1)[0][0]
    confidence = votes.most_common(1)[0][1] / k
    
    return prediction, confidence, votes

def train_test_split(data, test_ratio=0.2):
    """Split data into training and testing sets"""
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_ratio))
    return data[:split_index], data[split_index:]

def evaluate_model(train_data, test_data):
    """Evaluate the simple model"""
    print("\n=== MODEL EVALUATION ===")
    correct_predictions = 0
    total_predictions = len(test_data)
    
    predictions = []
    actual_labels = []
    
    for test_sample in test_data:
        prediction, confidence, votes = simple_knn_classifier(train_data, test_sample)
        predictions.append(prediction)
        actual_labels.append(test_sample[-1])
        
        if prediction == test_sample[-1]:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    print(f"Model Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    
    # Confusion matrix (simplified)
    crop_types = list(set(actual_labels))
    print(f"\nTested on {len(crop_types)} different crops")
    
    return accuracy

def crop_recommendation_interface(train_data):
    """Interactive crop recommendation interface"""
    print("\n=== CROP RECOMMENDATION SYSTEM ===")
    print("Enter soil and environmental parameters:")
    
    try:
        N = float(input("Nitrogen (N): "))
        P = float(input("Phosphorus (P): "))
        K = float(input("Potassium (K): "))
        temperature = float(input("Temperature (°C): "))
        humidity = float(input("Humidity (%): "))
        ph = float(input("pH level: "))
        rainfall = float(input("Rainfall (mm): "))
        
        user_sample = [N, P, K, temperature, humidity, ph, rainfall, None]
        
        # Get recommendation
        prediction, confidence, votes = simple_knn_classifier(train_data, user_sample, k=7)
        
        print(f"\n--- CROP RECOMMENDATION ---")
        print(f"Recommended Crop: {prediction}")
        print(f"Confidence: {confidence:.2f} ({int(confidence*100)}%)")
        
        print(f"\nTop recommendations based on similar conditions:")
        for i, (crop, count) in enumerate(votes.most_common(3), 1):
            percentage = (count / 7) * 100
            print(f"{i}. {crop} ({percentage:.1f}%)")
        
    except ValueError:
        print("Please enter valid numeric values.")
    except KeyboardInterrupt:
        print("\nExiting...")

def create_powerbi_data(data, header):
    """Create visualization data for PowerBI"""
    print("\n=== CREATING POWERBI VISUALIZATION DATA ===")
    
    # Crop distribution
    crops = [row[-1] for row in data]
    crop_counts = Counter(crops)
    
    # Create separate files for different visualizations
    
    # 1. Crop distribution data
    crop_dist_data = []
    for crop, count in crop_counts.items():
        crop_dist_data.append({
            'crop': crop,
            'count': count,
            'percentage': round((count / len(data)) * 100, 2)
        })
    
    with open('visualizations/crop_distribution.csv', 'w', newline='') as file:
        if crop_dist_data:
            writer = csv.DictWriter(file, fieldnames=['crop', 'count', 'percentage'])
            writer.writeheader()
            writer.writerows(crop_dist_data)
    
    # 2. Average conditions per crop
    crop_conditions = defaultdict(list)
    for row in data:
        crop = row[-1]
        crop_conditions[crop].append(row[:-1])  # Exclude label
    
    conditions_data = []
    for crop, conditions_list in crop_conditions.items():
        for i, param_name in enumerate(header[:-1]):  # Exclude label column
            avg_value = sum(row[i] for row in conditions_list) / len(conditions_list)
            conditions_data.append({
                'crop': crop,
                'parameter': param_name,
                'average_value': round(avg_value, 2)
            })
    
    with open('visualizations/crop_conditions.csv', 'w', newline='') as file:
        if conditions_data:
            writer = csv.DictWriter(file, fieldnames=['crop', 'parameter', 'average_value'])
            writer.writeheader()
            writer.writerows(conditions_data)
    
    print("✓ PowerBI visualization data saved:")
    print("  - visualizations/crop_distribution.csv")
    print("  - visualizations/crop_conditions.csv")
    return crop_dist_data, conditions_data

def main():
    """Main function"""
    print("CROP ADVISORY SYSTEM - SIMPLE VERSION")
    print("=" * 50)
    
    # Load or create dataset
    data, header = load_dataset()
    
    # Analyze dataset
    analyze_dataset(data, header)
    
    # Split data for training and testing
    train_data, test_data = train_test_split(data)
    
    # Evaluate model
    accuracy = evaluate_model(train_data, test_data)
    
    # Create PowerBI visualization data
    powerbi_data = create_powerbi_data(data, header)
    
    # Interactive recommendation system
    while True:
        try:
            continue_rec = input("\nWould you like to get crop recommendations? (y/n): ")
            if continue_rec.lower() != 'y':
                break
            crop_recommendation_interface(train_data)
        except KeyboardInterrupt:
            break
    
    print("\n✓ Crop Advisory System completed!")
    print("Generated files:")
    print("- data/crop_recommendation.csv")
    print("- visualizations/powerbi_data.csv")

if __name__ == "__main__":
    main()