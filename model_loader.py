"""
model_loader.py
Model Utility for EV Battery Failure Classification

Loads strictly one single 'best_ev_battery_model.pkl' created by EV_Battery_Failure.ipynb.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_model_path():
    """Returns path to best_ev_battery_model.pkl in models/ or root."""
    subfolder_path = os.path.join("models", "best_ev_battery_model.pkl")
    if os.path.exists(subfolder_path):
        return subfolder_path
    return "best_ev_battery_model.pkl"


def get_dataset_path():
    """Returns path to ev_battery_health_subset.csv in data/ or root."""
    subfolder_path = os.path.join("data", "ev_battery_health_subset.csv")
    if os.path.exists(subfolder_path):
        return subfolder_path
    return "ev_battery_health_subset.csv"


# Default Feature definitions matching EV_Battery_Failure.ipynb
NUMERICAL_FEATURES = [
    "capacity_loss_percent", "cell_voltage_std", "odometer_km",
    "thermal_runaway_risk", "cycle_count", "internal_resistance",
    "vehicle_age_years", "battery_stress_index", "battery_capacity_kwh"
]
CATEGORICAL_FEATURES = ["battery_chemistry", "vehicle_brand", "vehicle_type"]


def is_model_available(path: str = None) -> bool:
    """Checks whether best_ev_battery_model.pkl exists."""
    target_path = path or get_model_path()
    return os.path.exists(target_path)


def load_model(path: str = None):
    """
    Attempts to load the ML model artifact from single best_ev_battery_model.pkl.
    
    Returns:
        tuple: (artifact_or_model, is_available: bool)
    """
    target_path = path or get_model_path()
    if not os.path.exists(target_path):
        return None, False

    try:
        artifact = joblib.load(target_path)
        return artifact, True
    except Exception as e:
        print(f"Error loading model from {target_path}: {e}")
        return None, False


def predict_failure(model_or_artifact, input_df: pd.DataFrame, reference_df: pd.DataFrame = None):
    """
    Executes prediction using single best_ev_battery_model.pkl artifact.
    Extracts model, scaler, and feature definitions from single pickle file.
    """
    if model_or_artifact is None:
        raise ValueError("Model is not loaded.")

    # Unpack single pickle dictionary or fallback to raw model
    if isinstance(model_or_artifact, dict):
        model = model_or_artifact.get("model")
        scaler = model_or_artifact.get("scaler")
        numerical_features = model_or_artifact.get("numerical_features", NUMERICAL_FEATURES)
        categorical_features = model_or_artifact.get("categorical_features", CATEGORICAL_FEATURES)
        target_columns = model_or_artifact.get("trained_columns")
    else:
        model = model_or_artifact
        scaler = None
        numerical_features = NUMERICAL_FEATURES
        categorical_features = CATEGORICAL_FEATURES
        target_columns = None

    if model is None:
        raise ValueError("Invalid model artifact inside best_ev_battery_model.pkl")

    # Load default dataset as reference for fallback column matching if needed
    if target_columns is None:
        dataset_path = get_dataset_path()
        if reference_df is None and os.path.exists(dataset_path):
            reference_df = pd.read_csv(dataset_path)

        if reference_df is not None:
            ref_X = reference_df[numerical_features + categorical_features]
            ref_encoded = pd.get_dummies(ref_X, columns=categorical_features, drop_first=True)
            target_columns = ref_encoded.columns

    # Fit fallback scaler if not bundled in single pickle
    if scaler is None:
        dataset_path = get_dataset_path()
        if reference_df is None and os.path.exists(dataset_path):
            reference_df = pd.read_csv(dataset_path)
        if reference_df is not None:
            scaler = StandardScaler()
            scaler.fit(reference_df[numerical_features].fillna(reference_df[numerical_features].median()))

    # Copy input DataFrame
    input_X = input_df[[c for c in numerical_features + categorical_features if c in input_df.columns]].copy()
    
    # One-Hot Encoding on categorical columns
    input_encoded = pd.get_dummies(input_X, columns=categorical_features, drop_first=True)

    if target_columns is not None:
        input_encoded = input_encoded.reindex(columns=target_columns, fill_value=0)

    input_encoded = input_encoded.fillna(0)

    # Scale numerical features
    if scaler is not None:
        existing_num = [c for c in numerical_features if c in input_encoded.columns]
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
