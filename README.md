# Electricity Consumption Forecasting Project

A machine learning project to forecast household electricity consumption using time-series data and multiple regression models.

## 📊 Project Overview

This project analyzes and forecasts electricity consumption patterns using household power consumption data. It implements multiple ML models with proper time-series handling and provides comprehensive evaluation metrics.

## 🎯 Key Features

- **Fixed Data Pipeline**: Corrected datetime parsing and consolidated data cleaning
- **Time-Series Aware**: Chronological train/test split (not random) for proper forecasting
- **Multiple Models**: Random Forest, XGBoost, Decision Tree, Linear Regression
- **Feature Engineering**: Lag features, rolling statistics, time-based features
- **Comprehensive Evaluation**: MAE, RMSE, R², feature importance, drift detection
- **Production Ready**: Saved models, reproducible environment, automated pipeline

## 📁 Project Structure

```
d:\electric\
├── Electricity_Consumption_Forecasting_CLEANED.ipynb  # Main notebook (fixed version)
├── Electricity Consumption Forecasting Project (1) (1).ipynb  # Original notebook
├── run_pipeline.py                    # Automated pipeline script
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── test_pipeline.py                   # Unit tests
├── household_power_consumption.csv    # Dataset (you need to add this)
└── best_model.joblib                  # Saved model (generated after running)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Dataset: `household_power_consumption.csv` (UCI Machine Learning Repository)

### Installation

1. **Clone or navigate to the project directory:**
   ```powershell
   cd d:\electric
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Add the dataset:**
   - Download `household_power_consumption.csv` from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
   - Place it in `d:\electric\` directory

### Running the Project

**Option 1: Run the Jupyter Notebook**
```powershell
jupyter notebook Electricity_Consumption_Forecasting_CLEANED.ipynb
```
Then run all cells sequentially.

**Option 2: Run the Python Script**
```powershell
python run_pipeline.py
```

**Option 3: Run Tests**
```powershell
pytest test_pipeline.py -v
```

## 📈 Dataset Information

**Source**: UCI Machine Learning Repository - Individual Household Electric Power Consumption

**Features**:
- `Date` & `Time`: Measurement timestamp
- `Global_active_power`: Household active power (kilowatts)
- `Global_reactive_power`: Household reactive power (kilowatts)
- `Voltage`: Average voltage (volts)
- `Global_intensity`: Average current intensity (amperes)
- `Sub_metering_1, 2, 3`: Energy sub-metering (watt-hours)

**Target Variable**: `Global_active_power`

**Measurements**: Minute-level readings over several years

## 🔧 Key Fixes from Original Notebook

1. **Fixed Datetime Parsing**: Removed invalid `format='mixed'` parameter
2. **Chronological Split**: Replaced random train_test_split with time-ordered split
3. **Fixed Prediction Bugs**: Corrected shape mismatches in DecisionTree predictions
4. **Consolidated Cleaning**: Unified data cleaning into single section
5. **Added Feature Engineering**: Lag features, rolling stats, weekend indicators
6. **Added Diagnostics**: Dataset age check, drift detection, residual analysis
7. **Reproducible Environment**: Created requirements.txt, removed inline pip installs

## 📊 Model Performance

Models are evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

Typical results (depends on data):
- Random Forest: RMSE ~0.3-0.5 kW
- XGBoost: RMSE ~0.3-0.5 kW
- Decision Tree: RMSE ~0.4-0.6 kW
- Linear Regression: RMSE ~0.5-0.7 kW

## 🔍 Feature Engineering

**Time Features**:
- hour, day, month, day_of_week, is_weekend

**Lag Features**:
- lag_1h: Consumption 1 hour ago
- lag_24h: Consumption 24 hours ago

**Rolling Statistics**:
- rolling_mean_3h: 3-hour rolling mean
- rolling_std_3h: 3-hour rolling standard deviation

## 📝 Usage Examples

### Load Saved Model and Make Predictions

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('best_model.joblib')

# Prepare features (example)
features = pd.DataFrame({
    'Global_reactive_power': [0.1],
    'Voltage': [240.5],
    'Global_intensity': [1.5],
    'Sub_metering_1': [0.0],
    'Sub_metering_2': [1.0],
    'Sub_metering_3': [0.0],
    'hour': [14],
    'day': [15],
    'month': [6],
    'day_of_week': [2],
    'is_weekend': [0],
    'lag_1h': [0.5],
    'lag_24h': [0.6],
    'rolling_mean_3h': [0.55],
    'rolling_std_3h': [0.1]
})

# Predict
prediction = model.predict(features)
print(f"Predicted power consumption: {prediction[0]:.3f} kW")
```

## ⚠️ Important Notes

1. **Dataset Required**: You must add `household_power_consumption.csv` to run the project
2. **Time-Series Nature**: Always use chronological splits, never random shuffle for forecasting
3. **Data Recency**: Check dataset age - model may need retraining if data is >1 year old
4. **Feature Availability**: Ensure lag features are available at prediction time
5. **Computational Time**: Training on full dataset can take 5-15 minutes

## 🔄 Workflow

1. **Data Loading**: Read CSV, handle missing values ('?')
2. **Cleaning**: Convert types, drop NaNs, create datetime index
3. **Feature Engineering**: Create time features, lags, rolling stats
4. **Split**: 80% train (earlier data) / 20% test (later data)
5. **Training**: Train Random Forest, XGBoost, Decision Tree, Linear Regression
6. **Evaluation**: Compare models using MAE, RMSE, R²
7. **Selection**: Save best model
8. **Diagnostics**: Check for drift, analyze residuals

## 📚 Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- joblib >= 1.3.0

## 🧪 Testing

Run unit tests:
```powershell
pytest test_pipeline.py -v
```

Tests cover:
- Data loading and cleaning
- Datetime parsing
- Feature engineering
- Model training
- Prediction shape validation

## 🚀 Next Steps / Future Improvements

1. **Add Weather Data**: Temperature, humidity can improve accuracy
2. **Holiday Features**: Public holidays affect consumption patterns
3. **Hyperparameter Tuning**: GridSearchCV with TimeSeriesSplit
4. **Deep Learning**: Try LSTM/GRU for sequential patterns
5. **Deployment**: Create REST API with FastAPI
6. **Monitoring**: Add drift detection and auto-retraining
7. **Real-time Pipeline**: Stream processing for live predictions

## 📖 References

- Dataset: [UCI ML Repository - Household Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- Paper: Hebrail, G., & Berard, A. (2012). Individual household electric power consumption data set.

## 📄 License

This project is for educational purposes. Dataset license follows UCI ML Repository terms.

## 👤 Author

Created as part of an electricity consumption forecasting study.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add more features (weather, holidays)
- Implement LSTM models
- Create deployment pipeline
- Add more comprehensive tests
- Optimize hyperparameters

## 📞 Support

For issues or questions:
1. Check that `household_power_consumption.csv` is in the correct directory
2. Verify all dependencies are installed: `pip list`
3. Check Python version: `python --version` (should be 3.8+)

---

**Last Updated**: November 2025
