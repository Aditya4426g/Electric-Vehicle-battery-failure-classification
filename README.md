---
title: EV Battery Failure Classifier
colorFrom: red
colorTo: red
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# EV Battery Failure Classification & Health Monitoring Web Application

Predictive analytics and failure classification dashboard for Electric Vehicle (EV) battery packs using machine learning.

## Overview
This web application provides binary classification (Healthy vs. Failed) to predict EV battery pack degradation and failure risk based on vehicle telemetry and operational stress metrics.

## Key Features & Pages
- **Home**: Executive overview, problem rationale, and key failure drivers.
- **Battery Failure Prediction**: Live risk classification powered by `best_ev_battery_model.pkl` with interactive sliders.
- **Model Performance**: Evaluation metrics (93.00% Accuracy, 93.25% Recall, 0.9807 ROC-AUC), Confusion Matrix, ROC Curves, and Feature Coefficient Importance bar chart.
- **EDA Dashboard**: Target distribution bar plot, categorical count plots, numerical box plots, and correlation matrix heatmap.

## Machine Learning Pipeline
- **Champion Algorithm**: Tuned Logistic Regression (`C=0.1`).
- **Features (12 Total)**:
  - 9 Numerical: `capacity_loss_percent`, `cell_voltage_std`, `odometer_km`, `thermal_runaway_risk`, `cycle_count`, `internal_resistance`, `vehicle_age_years`, `battery_stress_index`, `battery_capacity_kwh`.
  - 3 Categorical: `battery_chemistry`, `vehicle_brand`, `vehicle_type`.

## Local Installation & Quickstart
```bash
# 1. Clone repository
git clone <your-repo-url>
cd p2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Application
streamlit run app.py
```

## Hugging Face Spaces Deployment
To deploy this application to Hugging Face Spaces:
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Select **Streamlit** as the Space SDK.
3. Upload all repository files (`app.py`, `model_loader.py`, `best_ev_battery_model.pkl`, `ev_battery_health_subset.csv`, `.streamlit/config.toml`, `requirements.txt`, `README.md`).
4. Hugging Face will automatically build and launch your live web app.
