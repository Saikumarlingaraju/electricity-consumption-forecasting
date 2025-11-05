# 🚀 QUICK START GUIDE

## Complete This Project in 3 Steps

### Step 1: Get the Dataset (2 minutes)

**Option A - Automated Download (Recommended)**
```powershell
python download_data.py
```

**Option B - Manual Download**
1. Visit: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
2. Download `household_power_consumption.zip`
3. Extract and rename to `household_power_consumption.csv`
4. Place in `d:\electric\` directory

### Step 2: Install Dependencies (1 minute)
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Project (Choose One)

**Option A - Run Full Pipeline Script (5-15 minutes)**
```powershell
python run_pipeline.py
```
This will:
- Load and clean data
- Engineer features
- Train 4 models (Random Forest, XGBoost, Decision Tree, Linear Regression)
- Evaluate and compare models
- Save the best model

**Option B - Run Jupyter Notebook (Interactive)**
```powershell
jupyter notebook Electricity_Consumption_Forecasting_CLEANED.ipynb
```
Then run all cells (Cell → Run All)

**Option C - Run Tests First (Optional)**
```powershell
pytest test_pipeline.py -v
```

---

## 📊 What You'll Get

After running, you'll have:
- ✅ Cleaned and processed dataset
- ✅ 4 trained models with performance metrics
- ✅ `best_model.joblib` - saved model for predictions
- ✅ Visualizations of predictions, errors, feature importance
- ✅ Drift detection analysis
- ✅ Performance comparison table

---

## 🎯 Expected Results

**Typical Output:**
```
ELECTRICITY CONSUMPTION FORECASTING PIPELINE
============================================================
Loading data...
✅ Data loaded and cleaned: (2049280, 7)
Creating datetime index...
✅ DateTime index created: 2006-12-16 to 2010-11-26
Engineering features...
✅ Features engineered: (2047920, 15)
Preparing train/test split...
✅ Train: 1,638,336 samples, Test: 409,584 samples

Training models...
  - Training Random Forest...
  - Training XGBoost...
✅ All models trained

Evaluating models...
  RandomForest: MAE=0.2847, RMSE=0.4531, R²=0.9234
  XGBoost: MAE=0.2719, RMSE=0.4398, R²=0.9279

✅ Best model (XGBoost) saved to 'best_model.joblib'

============================================================
PIPELINE COMPLETE
============================================================
Best Model: XGBoost
Best RMSE: 0.4398 kW
Features used: 15
============================================================
```

---

## ⚡ Quick Commands Reference

| Task | Command |
|------|---------|
| Download data | `python download_data.py` |
| Install packages | `pip install -r requirements.txt` |
| Run pipeline | `python run_pipeline.py` |
| Run notebook | `jupyter notebook Electricity_Consumption_Forecasting_CLEANED.ipynb` |
| Run tests | `pytest test_pipeline.py -v` |
| Check installed packages | `pip list` |
| Python version | `python --version` |

---

## 🔧 Troubleshooting

**Problem: Dataset not found**
```
FileNotFoundError: household_power_consumption.csv
```
**Solution:** Run `python download_data.py` or manually download the dataset

**Problem: Module not found**
```
ModuleNotFoundError: No module named 'xgboost'
```
**Solution:** Run `pip install -r requirements.txt`

**Problem: Memory error**
```
MemoryError
```
**Solution:** The dataset is ~2M rows. If you have <4GB RAM, modify `run_pipeline.py` to use a sample:
```python
df = df.sample(n=100000, random_state=42)  # Use 100k rows
```

**Problem: Jupyter not installed**
```
'jupyter' is not recognized
```
**Solution:** `pip install jupyter notebook`

---

## 📈 Next Steps After Completion

1. **Explore the Cleaned Notebook**
   - Open `Electricity_Consumption_Forecasting_CLEANED.ipynb`
   - Review visualizations and metrics
   - Experiment with different features

2. **Make Predictions**
   ```python
   import joblib
   model = joblib.load('best_model.joblib')
   # Use model.predict(new_data)
   ```

3. **Improve the Model**
   - Add weather data (temperature)
   - Add holiday features
   - Tune hyperparameters
   - Try LSTM/GRU models

4. **Deploy (Optional)**
   - Create REST API with FastAPI
   - Set up monitoring
   - Implement auto-retraining

---

## 📞 Still Stuck?

Check these in order:
1. ✅ Python 3.8+ installed: `python --version`
2. ✅ In correct directory: `cd d:\electric`
3. ✅ Dataset exists: `dir household_power_consumption.csv`
4. ✅ Dependencies installed: `pip list | findstr "xgboost"`
5. ✅ No errors in test: `pytest test_pipeline.py`

If all checks pass, you're ready to run!

---

**Time to Complete**: ~20 minutes total (download + install + run)

**Ready? Start with Step 1 above! 🚀**
