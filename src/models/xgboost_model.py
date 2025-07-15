import os
import joblib
import xgboost as xgb
import pandas as pd

def load_model_and_scaler(horizon: str):
    """
    Load the trained XGBoost model and corresponding scaler for the given horizon.

    Parameters:
    - horizon (str): Forecast horizon, e.g., "7d", "15d"

    Returns:
    - model: Trained XGBoost model
    - scaler: Trained StandardScaler
    """
    base_path = os.path.join("models", "xgboost", horizon)
    
    model_path = os.path.join(base_path, "model.pkl") 
    scaler_path = os.path.join(base_path, "scaler.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler

def predict_with_model(model, scaler, df_raw: pd.DataFrame, drop_cols=None):
    """
    Predict using all features in the DataFrame except the specified columns.

    Parameters:
    - model: Trained XGBoost model
    - scaler: StandardScaler object used during training
    - df_raw: Input DataFrame for prediction
    - drop_cols: Columns to exclude before prediction (e.g., labels)

    Returns:
    - y_pred: Predicted values
    """
    if drop_cols is None:
        drop_cols = []

    X = df_raw.drop(columns=drop_cols, errors='ignore')
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return y_pred

