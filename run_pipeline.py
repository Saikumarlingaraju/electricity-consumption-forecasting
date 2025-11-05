"""
Electricity Consumption Forecasting Pipeline
Main script to run the complete forecasting workflow
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_and_clean_data(filepath='household_power_consumption.csv'):
    """Load and clean the electricity consumption data"""
    print("Loading data...")
    
    # Try loading with semicolon separator first (common in this dataset)
    try:
        df = pd.read_csv(filepath, sep=';', low_memory=False)
        print(f"✅ Loaded with semicolon separator")
    except:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"✅ Loaded with comma separator")
    
    print(f"Columns found: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)
    
    # Convert numeric columns (handle both column name formats)
    cols_to_convert = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop missing values
    df.dropna(inplace=True)
    
    print(f"✅ Data loaded and cleaned: {df.shape}")
    return df


def create_datetime_index(df):
    """Create datetime index from Date and Time columns"""
    print("Creating datetime index...")
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    df.drop(columns=['Date', 'Time', 'index'], errors='ignore', inplace=True)
    
    print(f"✅ DateTime index created: {df.index.min()} to {df.index.max()}")
    return df


def engineer_features(df):
    """Engineer time-based and lag features"""
    print("Engineering features...")
    
    # Time features
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Lag features
    df['lag_1h'] = df['Global_active_power'].shift(60)
    df['lag_24h'] = df['Global_active_power'].shift(60*24)
    
    # Rolling statistics
    df['rolling_mean_3h'] = df['Global_active_power'].rolling(window=180, min_periods=1).mean()
    df['rolling_std_3h'] = df['Global_active_power'].rolling(window=180, min_periods=1).std()
    
    df.dropna(inplace=True)
    
    print(f"✅ Features engineered: {df.shape}")
    return df


def prepare_train_test_split(df, train_ratio=0.8):
    """Create chronological train/test split"""
    print("Preparing train/test split...")
    
    feature_cols = ['Global_reactive_power', 'Voltage', 'Global_intensity',
                    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
                    'hour', 'day', 'month', 'day_of_week', 'is_weekend',
                    'lag_1h', 'lag_24h', 'rolling_mean_3h', 'rolling_std_3h']
    
    X = df[feature_cols]
    y = df['Global_active_power']
    
    # Chronological split
    split_idx = int(len(df) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"✅ Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    return X_train, X_test, y_train, y_test, feature_cols


def train_models(X_train, y_train):
    """Train multiple models"""
    print("\nTraining models...")
    
    models = {}
    
    # Random Forest
    print("  - Training Random Forest...")
    models['RandomForest'] = RandomForestRegressor(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    models['RandomForest'].fit(X_train, y_train)
    
    # XGBoost
    print("  - Training XGBoost...")
    models['XGBoost'] = XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42, n_jobs=-1
    )
    models['XGBoost'].fit(X_train, y_train)
    
    print("✅ All models trained")
    return models


def evaluate_models(models, X_test, y_test):
    """Evaluate all models and return results"""
    print("\nEvaluating models...")
    
    results = []
    predictions = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        })
        
        print(f"  {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    
    results_df = pd.DataFrame(results).sort_values('RMSE')
    return results_df, predictions


def save_best_model(models, results_df, output_path='best_model.joblib'):
    """Save the best performing model"""
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    
    joblib.dump(best_model, output_path)
    print(f"\n✅ Best model ({best_model_name}) saved to '{output_path}'")
    
    return best_model_name, best_model


def main():
    """Main pipeline execution"""
    print("="*60)
    print("ELECTRICITY CONSUMPTION FORECASTING PIPELINE")
    print("="*60)
    
    # Load and preprocess data
    df = load_and_clean_data('household_power_consumption.csv')
    df = create_datetime_index(df)
    df = engineer_features(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(df)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results_df, predictions = evaluate_models(models, X_test, y_test)
    
    # Save best model
    best_model_name, best_model = save_best_model(models, results_df)
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(f"Best RMSE: {results_df.iloc[0]['RMSE']:.4f} kW")
    print(f"Features used: {len(feature_cols)}")
    print("="*60)
    
    return models, results_df, predictions


if __name__ == "__main__":
    models, results, predictions = main()
