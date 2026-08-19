"""
model_loader.py
Model Utility for EV Battery Failure Classification

Loads strictly 'best_ev_battery_model.pkl' created by EV_Battery_Failure.ipynb
and automatically scales numerical features using dataset statistics.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


MODEL_PATH = "best_ev_battery_model.pkl"

# Feature definitions matching EV_Battery_Failure.ipynb
NUMERICAL_FEATURES = [
    "capacity_loss_percent", "cell_voltage_std", "odometer_km",
    "thermal_runaway_risk", "cycle_count", "internal_resistance",
    "vehicle_age_years", "battery_stress_index", "battery_capacity_kwh"
]
CATEGORICAL_FEATURES = ["battery_chemistry", "vehicle_brand", "vehicle_type"]

# Cached fitted scaler in memory
_CACHED_SCALER = None


def is_model_available(path: str = MODEL_PATH) -> bool:
    """Checks whether best_ev_battery_model.pkl exists."""
    return os.path.exists(path)


def load_model(path: str = MODEL_PATH):
    """
    Attempts to load the ML model from best_ev_battery_model.pkl.
    
    Returns:
        tuple: (model_object, is_available: bool)
    """
    if not os.path.exists(path):
        return None, False

    try:
        model = joblib.load(path)
        return model, True
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        return None, False


def _get_scaler(reference_df: pd.DataFrame):
    """Returns or fits standard scaler in memory."""
    global _CACHED_SCALER
    if _CACHED_SCALER is None and reference_df is not None:
        _CACHED_SCALER = StandardScaler()
        _CACHED_SCALER.fit(reference_df[NUMERICAL_FEATURES].fillna(reference_df[NUMERICAL_FEATURES].median()))
    return _CACHED_SCALER


def predict_failure(model, input_df: pd.DataFrame, reference_df: pd.DataFrame = None):
    """
    Executes prediction using best_ev_battery_model.pkl.
    Preprocesses input numerical features with StandardScaler and one-hot encodes categorical attributes.
    """
    if model is None:
        raise ValueError("Model is not loaded.")

    # Load default dataset as reference if not provided
    if reference_df is None and os.path.exists("ev_battery_health_subset.csv"):
        reference_df = pd.read_csv("ev_battery_health_subset.csv")

    if reference_df is not None:
        ref_X = reference_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        ref_encoded = pd.get_dummies(ref_X, columns=CATEGORICAL_FEATURES, drop_first=True)
        target_columns = ref_encoded.columns
    else:
        target_columns = None

    scaler = _get_scaler(reference_df)

    # Copy input DataFrame
    input_X = input_df[[c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if c in input_df.columns]].copy()
    
    # One-Hot Encoding on categorical columns
    input_encoded = pd.get_dummies(input_X, columns=CATEGORICAL_FEATURES, drop_first=True)

    if target_columns is not None:
        input_encoded = input_encoded.reindex(columns=target_columns, fill_value=0)

    input_encoded = input_encoded.fillna(0)

    # Scale numerical features
    if scaler is not None:
        existing_num = [c for c in NUMERICAL_FEATURES if c in input_encoded.columns]
        input_encoded[existing_num] = scaler.transform(input_encoded[existing_num])

    # Predict class & failure probability
    prediction = int(model.predict(input_encoded)[0])
    probability = None

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(input_encoded)[0]
            probability = float(probs[1]) if len(probs) > 1 else float(probs[0])
        except Exception:
            probability = None

    return {
        "prediction": prediction,
        "probability": probability
    }
