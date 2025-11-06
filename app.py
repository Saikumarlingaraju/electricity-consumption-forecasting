"""
Streamlit App: Electricity Consumption Forecasting

Features:
- Evaluate the pre-trained model (best_model.joblib) on the dataset
- Browse and tweak samples from the test set to see predictions
- Batch prediction from uploaded CSV (expects raw UCI columns)

Notes:
- Reuses pipeline utilities from run_pipeline.py to ensure consistent preprocessing
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import StringIO

# Reuse pipeline functions
from run_pipeline import (
    load_and_clean_data,
    create_datetime_index,
    engineer_features,
    prepare_train_test_split,
)


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Electricity Forecasting", layout="wide")
st.title("⚡ Electricity Consumption Forecasting")
st.caption("Evaluate the trained model, explore predictions, and run batch inference.")


# -----------------------------
# Helpers & caching
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data_cached(filepath: str = 'household_power_consumption.csv'):
    df = load_and_clean_data(filepath)
    df = create_datetime_index(df)
    df = engineer_features(df)
    return df


@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str = 'best_model.joblib'):
    return joblib.load(model_path)


def evaluate_on_dataset(df: pd.DataFrame, model):
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(df)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    return {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_cols': feature_cols,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
    }


def render_metrics(mae: float, rmse: float, r2: float):
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (kW)", f"{mae:.4f}")
    col2.metric("RMSE (kW)", f"{rmse:.4f}")
    col3.metric("R²", f"{r2:.4f}")


def render_timeseries_plot(y_test: pd.Series, y_pred: np.ndarray, max_points: int = 10000):
    # Display only the tail for performance
    if len(y_test) > max_points:
        y_test_disp = y_test.iloc[-max_points:]
        y_pred_disp = y_pred[-max_points:]
    else:
        y_test_disp = y_test
        y_pred_disp = y_pred

    plot_df = pd.DataFrame({
        'Actual': y_test_disp.values,
        'Predicted': y_pred_disp,
    }, index=y_test_disp.index)
    st.line_chart(plot_df)


def require_columns(df: pd.DataFrame, required_cols: list[str]) -> tuple[bool, list[str]]:
    missing = [c for c in required_cols if c not in df.columns]
    return (len(missing) == 0, missing)


# -----------------------------
# Sidebar
# -----------------------------
mode = st.sidebar.radio(
    "Mode",
    [
        "Evaluate on dataset",
        "Browse test samples",
        "Batch predict (upload CSV)",
    ],
)

with st.sidebar:
    st.markdown("""
    ### Model
    This app loads the pre-trained model saved by the pipeline (`best_model.joblib`).
    """)


# -----------------------------
# Main views
# -----------------------------
model = load_model_cached('best_model.joblib')

if mode == "Evaluate on dataset":
    st.subheader("Evaluate on Dataset")
    with st.spinner("Loading and preprocessing data..."):
        df = load_data_cached('household_power_consumption.csv')

    results = evaluate_on_dataset(df, model)
    render_metrics(results['mae'], results['rmse'], results['r2'])

    st.markdown("#### Actual vs Predicted (last samples)")
    render_timeseries_plot(results['y_test'], results['y_pred'])

    # Detailed table (sampled for performance)
    st.markdown("#### Predictions table (sample)")
    sample_n = min(5000, len(results['y_test']))
    table_df = pd.DataFrame({
        'actual': results['y_test'].values[-sample_n:],
        'predicted': results['y_pred'][-sample_n:],
    }, index=results['y_test'].index[-sample_n:])
    st.dataframe(table_df)

    # Download predictions
    csv_buf = StringIO()
    out_full = pd.DataFrame({
        'timestamp': results['y_test'].index,
        'actual': results['y_test'].values,
        'predicted': results['y_pred'],
    })
    out_full.to_csv(csv_buf, index=False)
    st.download_button("Download full predictions CSV", csv_buf.getvalue(), file_name="predictions.csv", mime="text/csv")


elif mode == "Browse test samples":
    st.subheader("Browse and Tweak Test Samples")
    with st.spinner("Preparing data..."):
        df = load_data_cached('household_power_consumption.csv')
        X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(df)

    st.caption("Select a row from the test set, tweak features, and see the model's prediction.")
    idx = st.number_input("Test row index (relative to test set start)", min_value=0, max_value=max(0, len(X_test) - 1), value=0, step=1)

    row = X_test.iloc[int(idx)].copy()
    cols_left, cols_right = st.columns(2)

    with cols_left:
        for col in feature_cols[: len(feature_cols)//2]:
            val = float(row[col])
            row[col] = st.number_input(col, value=val)

    with cols_right:
        for col in feature_cols[len(feature_cols)//2 :]:
            val = float(row[col])
            row[col] = st.number_input(col, value=val)

    # Predict
    pred = model.predict(pd.DataFrame([row], columns=feature_cols))[0]
    st.markdown(f"### Predicted Global Active Power: `{pred:.4f} kW`")

    # Show actual if available
    actual = y_test.iloc[int(idx)]
    st.markdown(f"Actual (for selected row): `{actual:.4f} kW`")


elif mode == "Batch predict (upload CSV)":
    st.subheader("Batch Prediction via CSV Upload")
    st.markdown("""
    Upload raw UCI data with at least these columns: 
    `Date, Time, Global_active_power, Global_reactive_power, Voltage, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3`.

    The app will apply the same preprocessing and feature engineering as the pipeline, then produce predictions.
    """)

    uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
    if uploaded is not None:
        try:
            # Try semicolon first (dataset default), fall back to comma
            try:
                raw = pd.read_csv(uploaded, sep=';', low_memory=False)
            except Exception:
                uploaded.seek(0)
                raw = pd.read_csv(uploaded, low_memory=False)

            required_raw = [
                'Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage',
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
            ]
            ok, missing = require_columns(raw, required_raw)
            if not ok:
                st.error(f"Missing required columns: {missing}")
            else:
                # Process like pipeline
                df = raw.copy()
                df.replace('?', np.nan, inplace=True)
                for col in ['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.dropna(inplace=True)
                df = create_datetime_index(df)
                df = engineer_features(df)

                # Prepare features
                feature_cols = ['Global_reactive_power', 'Voltage', 'Global_intensity',
                                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
                                'hour', 'day', 'month', 'day_of_week', 'is_weekend',
                                'lag_1h', 'lag_24h', 'rolling_mean_3h', 'rolling_std_3h']

                X = df[feature_cols]
                preds = model.predict(X)

                st.success(f"Predictions generated for {len(X):,} rows")
                # Show sample table
                show_n = min(5000, len(X))
                preview = pd.DataFrame({
                    'timestamp': X.index[:show_n],
                    'predicted_global_active_power': preds[:show_n]
                })
                st.dataframe(preview)

                # Download full results
                out_buf = StringIO()
                out_df = pd.DataFrame({
                    'timestamp': X.index,
                    'predicted_global_active_power': preds,
                })
                out_df.to_csv(out_buf, index=False)
                st.download_button("Download predictions CSV", out_buf.getvalue(), file_name="batch_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")


st.divider()
st.caption("Model: loaded from best_model.joblib. Data source: household_power_consumption.csv")
