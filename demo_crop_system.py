#!/usr/bin/env python3
"""
Crop Advisory System - Demo Version
Demonstrates the complete system without interactive input
"""

import csv
import random
import math
from collections import defaultdict, Counter

def create_crop_dataset():
    """Create a comprehensive crop recommendation dataset"""
    random.seed(42)
    
    # Define crops with their optimal conditions based on real agricultural data
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
        for _ in range(250):  # 250 samples per crop for better model training
            sample = []
            for param in ['N', 'P', 'K', 'temp', 'humidity', 'ph', 'rainfall']:
                optimal = conditions[param]
                # Add controlled random variation (±15% of optimal value)
                noise_factor = random.uniform(-0.15, 0.15)
                value = max(0, optimal + (noise_factor * optimal))
                
                # Apply realistic constraints
                if param == 'ph':
                    value = max(3.5, min(9.5, value))  # Agricultural pH range
                elif param == 'humidity':
                    value = max(10, min(100, value))
                elif param == 'temp':
                    value = max(5, min(45, value))  # Agricultural temperature range
                
                sample.append(round(value, 2))
            
            sample.append(crop)
            data.append(sample)
    
    # Shuffle the data for better training
    random.shuffle(data)
    
    # Save to CSV
    with open('data/crop_recommendation.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)
    
    return data, header, crop_conditions

def simple_euclidean_distance(sample1, sample2):
    """Calculate normalized Euclidean distance between two samples"""
    distance = 0
    # Normalize by expected ranges to give equal weight to all parameters
    weights = [100, 100, 200, 30, 90, 5, 200]  # Approximate ranges for N,P,K,temp,humidity,pH,rainfall
    
    for i in range(len(sample1) - 1):  # Exclude label
        normalized_diff = (sample1[i] - sample2[i]) / weights[i]
        distance += normalized_diff ** 2
    return math.sqrt(distance)

def enhanced_knn_classifier(train_data, test_sample, k=7):
    """Enhanced K-Nearest Neighbors classifier with weighted voting"""
    distances = []
    
    for train_sample in train_data:
        dist = simple_euclidean_distance(train_sample, test_sample)
        distances.append((dist, train_sample[-1]))
    
    # Sort by distance and get k nearest neighbors
    distances.sort()
    nearest_neighbors = distances[:k]
    
    # Weighted voting (inverse distance weighting)
    weighted_votes = defaultdict(float)
    for dist, label in nearest_neighbors:
        # Add small epsilon to avoid division by zero
        weight = 1 / (dist + 0.001)
        weighted_votes[label] += weight
    
    # Get the prediction with highest weighted vote
    prediction = max(weighted_votes, key=weighted_votes.get)
    total_weight = sum(weighted_votes.values())
    confidence = weighted_votes[prediction] / total_weight
    
    return prediction, confidence, dict(weighted_votes)

def train_test_split(data, test_ratio=0.2):
    """Split data into training and testing sets with stratification"""
    # Group by crop type for stratified split
    crop_groups = defaultdict(list)
    for sample in data:
        crop_groups[sample[-1]].append(sample)
    
    train_data = []
    test_data = []
    
    for crop, samples in crop_groups.items():
        random.shuffle(samples)
        split_index = int(len(samples) * (1 - test_ratio))
        train_data.extend(samples[:split_index])
        test_data.extend(samples[split_index:])
    
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    return train_data, test_data

def evaluate_model(train_data, test_data):
    """Comprehensive model evaluation"""
    print("\n=== MODEL EVALUATION ===")
    correct_predictions = 0
    crop_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for test_sample in test_data:
        prediction, confidence, votes = enhanced_knn_classifier(train_data, test_sample)
        actual_crop = test_sample[-1]
        
        crop_accuracy[actual_crop]['total'] += 1
        if prediction == actual_crop:
            correct_predictions += 1
            crop_accuracy[actual_crop]['correct'] += 1
    
    overall_accuracy = correct_predictions / len(test_data)
    print(f"Overall Model Accuracy: {overall_accuracy:.4f} ({correct_predictions}/{len(test_data)})")
    
    print("\nPer-Crop Accuracy:")
    for crop in sorted(crop_accuracy.keys()):
        acc = crop_accuracy[crop]['correct'] / crop_accuracy[crop]['total']
        print(f"  {crop}: {acc:.3f} ({crop_accuracy[crop]['correct']}/{crop_accuracy[crop]['total']})")
    
    return overall_accuracy

def create_powerbi_visualizations(data, header, crop_conditions):
    """Create comprehensive visualization data for PowerBI"""
    print("\n=== CREATING POWERBI VISUALIZATION DATA ===")
    
    crops = [row[-1] for row in data]
    crop_counts = Counter(crops)
    
    # 1. Crop Distribution
    crop_dist_data = []
    for crop, count in crop_counts.items():
        crop_dist_data.append({
            'crop': crop,
            'count': count,
            'percentage': round((count / len(data)) * 100, 2)
        })
    
    with open('visualizations/crop_distribution.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['crop', 'count', 'percentage'])
        writer.writeheader()
        writer.writerows(crop_dist_data)
    
    # 2. Average Conditions per Crop
    crop_conditions_data = defaultdict(list)
    for row in data:
        crop = row[-1]
        crop_conditions_data[crop].append(row[:-1])
    
    conditions_data = []
    for crop, conditions_list in crop_conditions_data.items():
        for i, param_name in enumerate(header[:-1]):
            avg_value = sum(row[i] for row in conditions_list) / len(conditions_list)
            conditions_data.append({
                'crop': crop,
                'parameter': param_name,
                'average_value': round(avg_value, 2)
            })
    
    with open('visualizations/crop_conditions.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['crop', 'parameter', 'average_value'])
        writer.writeheader()
        writer.writerows(conditions_data)
    
    # 3. Optimal vs Actual Conditions Comparison
    comparison_data = []
    for crop, optimal in crop_conditions.items():
        if crop in crop_conditions_data:
            actual_conditions = crop_conditions_data[crop]
            for i, param in enumerate(['N', 'P', 'K', 'temp', 'humidity', 'ph', 'rainfall']):
                optimal_val = optimal[param]
                actual_avg = sum(row[i] for row in actual_conditions) / len(actual_conditions)
                comparison_data.append({
                    'crop': crop,
                    'parameter': param,
                    'optimal_value': optimal_val,
                    'actual_average': round(actual_avg, 2),
                    'difference': round(actual_avg - optimal_val, 2)
                })
    
    with open('visualizations/optimal_vs_actual.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['crop', 'parameter', 'optimal_value', 'actual_average', 'difference'])
        writer.writeheader()
        writer.writerows(comparison_data)
    
    print("✓ PowerBI visualization files created:")
    print("  - visualizations/crop_distribution.csv")
    print("  - visualizations/crop_conditions.csv")
    print("  - visualizations/optimal_vs_actual.csv")

def demonstrate_predictions(train_data):
    """Demonstrate the system with sample predictions"""
    print("\n=== CROP RECOMMENDATION DEMONSTRATIONS ===")
    
    # Sample scenarios for demonstration
    test_scenarios = [
        {
            'name': 'High Nitrogen Rich Soil',
            'conditions': [120, 80, 60, 25, 70, 6.5, 100],
            'description': 'Suitable for nitrogen-loving crops'
        },
        {
            'name': 'Low Humidity Arid Region',
            'conditions': [60, 70, 100, 22, 20, 7.0, 40],
            'description': 'Dry climate with low rainfall'
        },
        {
            'name': 'High Rainfall Tropical',
            'conditions': [90, 50, 50, 28, 85, 6.0, 200],
            'description': 'Tropical climate with high humidity and rainfall'
        },
        {
            'name': 'Temperate Fruit Growing',
            'conditions': [30, 130, 200, 16, 80, 6.5, 80],
            'description': 'Cool climate with high P and K for fruit trees'
        }
    ]
    
    for scenario in test_scenarios:
        conditions = scenario['conditions'] + [None]  # Add placeholder for label
        prediction, confidence, votes = enhanced_knn_classifier(train_data, conditions)
        
        print(f"\nScenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Conditions: N={conditions[0]}, P={conditions[1]}, K={conditions[2]}, "
              f"Temp={conditions[3]}°C, Humidity={conditions[4]}%, pH={conditions[5]}, "
              f"Rainfall={conditions[6]}mm")
        print(f"Recommended Crop: {prediction} (Confidence: {confidence:.2f})")
        
        # Show top 3 recommendations
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)[:3]
        print("Top 3 Recommendations:")
        for i, (crop, weight) in enumerate(sorted_votes, 1):
            percentage = (weight / sum(votes.values())) * 100
            print(f"  {i}. {crop} ({percentage:.1f}%)")

def main():
    """Main function to run the complete crop advisory system"""
    print("CROP ADVISORY SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 60)
    print("Development of a Data-Driven Crop Advisory System")
    print("Using Soil and Environmental Parameters")
    print("=" * 60)
    
    # Create comprehensive dataset
    print("Creating comprehensive crop recommendation dataset...")
    data, header, crop_conditions = create_crop_dataset()
    print(f"✓ Dataset created with {len(data)} samples and {len(crop_conditions)} crops")
    
    # Dataset Analysis
    print(f"\n=== DATASET ANALYSIS ===")
    crops = [row[-1] for row in data]
    crop_counts = Counter(crops)
    print(f"Total samples: {len(data)}")
    print(f"Unique crops: {len(crop_counts)}")
    print("Crop distribution:")
    for crop, count in sorted(crop_counts.items()):
        print(f"  {crop}: {count} samples ({count/len(data)*100:.1f}%)")
    
    # Basic statistics
    print("\nParameter Statistics:")
    param_names = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    for i, param in enumerate(param_names):
        values = [row[i] for row in data]
        print(f"  {param}: Min={min(values):.1f}, Max={max(values):.1f}, "
              f"Mean={sum(values)/len(values):.1f}")
    
    # Model Training and Evaluation
    train_data, test_data = train_test_split(data, test_ratio=0.2)
    print(f"\nDataset split: {len(train_data)} training, {len(test_data)} testing samples")
    
    accuracy = evaluate_model(train_data, test_data)
    
    # Create PowerBI visualizations
    create_powerbi_visualizations(data, header, crop_conditions)
    
    # Demonstrate predictions
    demonstrate_predictions(train_data)
    
    print(f"\n{'='*60}")
    print("✓ CROP ADVISORY SYSTEM COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("Generated Files:")
    print("📊 data/crop_recommendation.csv - Main dataset (2,250 samples)")
    print("📈 visualizations/crop_distribution.csv - For PowerBI charts")
    print("📊 visualizations/crop_conditions.csv - Parameter analysis")
    print("📉 visualizations/optimal_vs_actual.csv - Comparison data")
    print(f"\nModel Performance: {accuracy:.1%} accuracy on test set")
    print("Ready for PowerBI integration and further analysis!")

if __name__ == "__main__":
    main()