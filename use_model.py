"""
Demo script: How to load and use the trained model for predictions
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*60)
print("ELECTRICITY CONSUMPTION PREDICTION DEMO")
print("="*60)

# Load the trained model
print("\n1. Loading the trained model...")
model = joblib.load('best_model.joblib')
print(f"✅ Model loaded: {type(model).__name__}")

# Example: Create sample data for prediction
print("\n2. Creating sample input data...")
print("   (In production, you would load real sensor data)")

# Feature names expected by the model
feature_cols = ['Global_reactive_power', 'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
                'hour', 'day', 'month', 'day_of_week', 'is_weekend',
                'lag_1h', 'lag_24h', 'rolling_mean_3h', 'rolling_std_3h']

# Create sample input (realistic values based on the dataset)
sample_input = pd.DataFrame({
    'Global_reactive_power': [0.123],
    'Voltage': [241.5],
    'Global_intensity': [5.2],
    'Sub_metering_1': [1.0],
    'Sub_metering_2': [1.0],
    'Sub_metering_3': [17.0],
    'hour': [19],  # 7 PM
    'day': [5],
    'month': [11],
    'day_of_week': [2],  # Wednesday
    'is_weekend': [0],
    'lag_1h': [1.2],  # Previous hour consumption
    'lag_24h': [1.3],  # Same time yesterday
    'rolling_mean_3h': [1.25],
    'rolling_std_3h': [0.15]
})

print("✅ Sample input created")
print(f"\n   Input features:\n{sample_input.T}")

# Make prediction
print("\n3. Making prediction...")
prediction = model.predict(sample_input)
print(f"✅ Predicted power consumption: {prediction[0]:.4f} kW")

# Batch prediction example
print("\n4. Batch prediction example (3 time points)...")
batch_input = pd.DataFrame({
    'Global_reactive_power': [0.123, 0.145, 0.098],
    'Voltage': [241.5, 239.2, 243.1],
    'Global_intensity': [5.2, 6.1, 4.8],
    'Sub_metering_1': [1.0, 2.0, 0.5],
    'Sub_metering_2': [1.0, 1.5, 0.8],
    'Sub_metering_3': [17.0, 18.5, 15.2],
    'hour': [19, 20, 21],
    'day': [5, 5, 5],
    'month': [11, 11, 11],
    'day_of_week': [2, 2, 2],
    'is_weekend': [0, 0, 0],
    'lag_1h': [1.2, 1.3, 1.4],
    'lag_24h': [1.3, 1.4, 1.5],
    'rolling_mean_3h': [1.25, 1.30, 1.35],
    'rolling_std_3h': [0.15, 0.16, 0.17]
})

batch_predictions = model.predict(batch_input)
print("✅ Batch predictions:")
for i, pred in enumerate(batch_predictions, 1):
    print(f"   Time point {i}: {pred:.4f} kW")

print("\n" + "="*60)
print("USAGE TIPS:")
print("="*60)
print("1. Always provide all 15 features in the correct order")
print("2. Lag features (lag_1h, lag_24h) require historical data")
print("3. Rolling statistics need at least 3 hours of data")
print("4. Use chronological data for best results")
print("5. Retrain model periodically with new data")
print("="*60)
