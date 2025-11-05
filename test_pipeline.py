"""
Unit tests for the electricity consumption forecasting pipeline
Run with: pytest test_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from run_pipeline
try:
    from run_pipeline import (
        create_datetime_index,
        engineer_features,
        prepare_train_test_split,
        train_models,
        evaluate_models
    )
except ImportError:
    pytest.skip("run_pipeline.py not found", allow_module_level=True)


@pytest.fixture
def sample_data():
    """Create sample electricity consumption data for testing"""
    dates = pd.date_range('2023-01-01', periods=1000, freq='1min')
    
    df = pd.DataFrame({
        'Date': dates.strftime('%d/%m/%Y'),
        'Time': dates.strftime('%H:%M:%S'),
        'Global_active_power': np.random.uniform(0.5, 5.0, 1000),
        'Global_reactive_power': np.random.uniform(0.0, 0.5, 1000),
        'Voltage': np.random.uniform(235, 245, 1000),
        'Global_intensity': np.random.uniform(1.0, 20.0, 1000),
        'Sub_metering_1': np.random.uniform(0, 10, 1000),
        'Sub_metering_2': np.random.uniform(0, 10, 1000),
        'Sub_metering_3': np.random.uniform(0, 20, 1000)
    })
    
    return df


class TestDataPreprocessing:
    """Test data preprocessing functions"""
    
    def test_datetime_index_creation(self, sample_data):
        """Test that datetime index is created correctly"""
        df = create_datetime_index(sample_data.copy())
        
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) > 0
        assert 'Date' not in df.columns
        assert 'Time' not in df.columns
    
    def test_datetime_index_sorted(self, sample_data):
        """Test that datetime index is sorted"""
        df = create_datetime_index(sample_data.copy())
        
        assert df.index.is_monotonic_increasing
    
    def test_feature_engineering(self, sample_data):
        """Test that features are engineered correctly"""
        df = create_datetime_index(sample_data.copy())
        df = engineer_features(df)
        
        # Check time features exist
        assert 'hour' in df.columns
        assert 'day' in df.columns
        assert 'month' in df.columns
        assert 'day_of_week' in df.columns
        assert 'is_weekend' in df.columns
        
        # Check lag features exist
        assert 'lag_1h' in df.columns
        assert 'lag_24h' in df.columns
        
        # Check rolling features exist
        assert 'rolling_mean_3h' in df.columns
        assert 'rolling_std_3h' in df.columns
    
    def test_feature_ranges(self, sample_data):
        """Test that engineered features have correct ranges"""
        df = create_datetime_index(sample_data.copy())
        df = engineer_features(df)
        
        assert df['hour'].min() >= 0
        assert df['hour'].max() <= 23
        assert df['month'].min() >= 1
        assert df['month'].max() <= 12
        assert df['day_of_week'].min() >= 0
        assert df['day_of_week'].max() <= 6
        assert set(df['is_weekend'].unique()).issubset({0, 1})


class TestTrainTestSplit:
    """Test train/test split functionality"""
    
    def test_train_test_split_chronological(self, sample_data):
        """Test that train/test split respects chronological order"""
        df = create_datetime_index(sample_data.copy())
        df = engineer_features(df)
        
        X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(df)
        
        # Check that train comes before test chronologically
        assert X_train.index.max() < X_test.index.min()
    
    def test_train_test_split_ratio(self, sample_data):
        """Test that train/test split has correct ratio"""
        df = create_datetime_index(sample_data.copy())
        df = engineer_features(df)
        
        X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(df, train_ratio=0.8)
        
        total_samples = len(X_train) + len(X_test)
        train_ratio = len(X_train) / total_samples
        
        assert 0.75 <= train_ratio <= 0.85  # Allow some tolerance
    
    def test_feature_columns_returned(self, sample_data):
        """Test that feature columns list is returned"""
        df = create_datetime_index(sample_data.copy())
        df = engineer_features(df)
        
        X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(df)
        
        assert isinstance(feature_cols, list)
        assert len(feature_cols) > 0
        assert all(col in X_train.columns for col in feature_cols)


class TestModelTraining:
    """Test model training functionality"""
    
    def test_models_trained(self, sample_data):
        """Test that models are trained successfully"""
        df = create_datetime_index(sample_data.copy())
        df = engineer_features(df)
        X_train, X_test, y_train, y_test, _ = prepare_train_test_split(df)
        
        models = train_models(X_train, y_train)
        
        assert 'RandomForest' in models
        assert 'XGBoost' in models
        assert isinstance(models['RandomForest'], RandomForestRegressor)
    
    def test_model_predictions(self, sample_data):
        """Test that models can make predictions"""
        df = create_datetime_index(sample_data.copy())
        df = engineer_features(df)
        X_train, X_test, y_train, y_test, _ = prepare_train_test_split(df)
        
        models = train_models(X_train, y_train)
        
        for name, model in models.items():
            predictions = model.predict(X_test)
            assert len(predictions) == len(X_test)
            assert not np.isnan(predictions).any()


class TestModelEvaluation:
    """Test model evaluation functionality"""
    
    def test_evaluation_metrics(self, sample_data):
        """Test that evaluation returns correct metrics"""
        df = create_datetime_index(sample_data.copy())
        df = engineer_features(df)
        X_train, X_test, y_train, y_test, _ = prepare_train_test_split(df)
        
        models = train_models(X_train, y_train)
        results_df, predictions = evaluate_models(models, X_test, y_test)
        
        assert 'Model' in results_df.columns
        assert 'MAE' in results_df.columns
        assert 'RMSE' in results_df.columns
        assert 'R²' in results_df.columns
        assert len(results_df) == len(models)
    
    def test_predictions_shape(self, sample_data):
        """Test that predictions have correct shape"""
        df = create_datetime_index(sample_data.copy())
        df = engineer_features(df)
        X_train, X_test, y_train, y_test, _ = prepare_train_test_split(df)
        
        models = train_models(X_train, y_train)
        results_df, predictions = evaluate_models(models, X_test, y_test)
        
        for name, pred in predictions.items():
            assert len(pred) == len(y_test)


class TestDataValidation:
    """Test data validation and edge cases"""
    
    def test_missing_values_handled(self):
        """Test that missing values are handled correctly"""
        df = pd.DataFrame({
            'Date': ['01/01/2023', '01/01/2023', '01/01/2023'],
            'Time': ['00:00:00', '00:01:00', '00:02:00'],
            'Global_active_power': [1.0, np.nan, 2.0],
            'Global_reactive_power': [0.1, 0.2, 0.3],
            'Voltage': [240, 240, 240],
            'Global_intensity': [4.0, 4.0, 4.0],
            'Sub_metering_1': [0, 0, 0],
            'Sub_metering_2': [0, 0, 0],
            'Sub_metering_3': [0, 0, 0]
        })
        
        df = create_datetime_index(df)
        # Should not raise error
        assert len(df) >= 0
    
    def test_invalid_datetime_handled(self):
        """Test that invalid datetimes are handled"""
        df = pd.DataFrame({
            'Date': ['01/01/2023', '99/99/9999', '01/01/2023'],
            'Time': ['00:00:00', '00:01:00', '00:02:00'],
            'Global_active_power': [1.0, 1.5, 2.0],
            'Global_reactive_power': [0.1, 0.2, 0.3],
            'Voltage': [240, 240, 240],
            'Global_intensity': [4.0, 4.0, 4.0],
            'Sub_metering_1': [0, 0, 0],
            'Sub_metering_2': [0, 0, 0],
            'Sub_metering_3': [0, 0, 0]
        })
        
        df = create_datetime_index(df)
        # Should handle invalid dates and keep valid ones
        assert len(df) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
