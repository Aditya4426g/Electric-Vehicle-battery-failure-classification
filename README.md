# EV Battery Failure Classification & Health Monitoring Web Application

Predictive analytics and failure classification dashboard for Electric Vehicle (EV) battery packs using machine learning.

---

## Overview

This application provides real-time binary classification (Healthy vs. High Risk Failure) to monitor EV battery degradation based on vehicle telemetry and operational stress parameters. Built using Streamlit, Python, and Scikit-Learn.

---

## Directory Structure

```text
Electric-Vehicle-battery-failure-classification/
├── data/
│   └── ev_battery_health_subset.csv       # Telemetry dataset (4,000 records)
├── models/
│   └── best_ev_battery_model.pkl          # Tuned Logistic Regression model
├── docs/
│   └── EV_Battery_Failure.ipynb           # Model development & EDA notebook
├── .streamlit/
│   └── config.toml                        # Streamlit theme & server configuration
├── app.py                                 # Streamlit Web Application entrypoint
├── model_loader.py                        # Preprocessing & inference pipeline
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
└── README.md                              # Project documentation
```

---

## Machine Learning Performance

* **Best Model Algorithm**: Tuned Logistic Regression (`C=0.1`)
* **Test Classification Accuracy**: **93.00%**
* **Failure Recall**: **93.25%** (Detects 373 out of 400 failing battery packs)
* **Discriminative Power**: **0.9807 ROC-AUC**
* **F1 Score**: **93.02%**

---

## Key Telemetry Features (12 Features)

### 9 Numerical Features:
1. `capacity_loss_percent` — State of Health loss (%)
2. `cell_voltage_std` — Cell voltage imbalance (V)
3. `odometer_km` — Total cumulative mileage (km)
4. `thermal_runaway_risk` — Thermal runaway risk index (0–100)
5. `cycle_count` — Full charge/discharge cycles
6. `internal_resistance` — Cell impedance (mΩ)
7. `vehicle_age_years` — Vehicle operational age (years)
8. `battery_stress_index` — Composite stress score (0–100)
9. `battery_capacity_kwh` — Pack energy rating (kWh)

### 3 Categorical Features:
1. `battery_chemistry` — Battery cell chemistry (NMC, LFP, NCA, LMO, LTO)
2. `vehicle_brand` — Automotive OEM brand
3. `vehicle_type` — Vehicle form factor (SUV, Sedan, Hatchback, Crossover, Truck, Van)

---

## Web Application Pages

1. **Home**: Executive summary, problem rationale, and key failure drivers.
2. **Battery Failure Prediction**: Live risk classification with interactive sliders powered by `best_ev_battery_model.pkl`.
3. **Model Performance**: Metrics summary, Confusion Matrix, ROC-AUC Curves, and Feature Coefficients bar chart.
4. **EDA Dashboard**: Target distribution bar plot, grouped countplots, numerical boxplots, and correlation matrix heatmap.

---

## Local Installation & Quickstart

```bash
# 1. Clone repository
git clone https://github.com/Aditya4426g/Electric-Vehicle-battery-failure-classification.git
cd Electric-Vehicle-battery-failure-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Application
streamlit run app.py
```

---

## Cloud Deployment (Streamlit Community Cloud)

This app is configured for 1-click free cloud hosting:

1. Sign in to [share.streamlit.io](https://share.streamlit.io/) with GitHub.
2. Click **New app** -> **Use existing repo**.
3. Select repository: `Aditya4426g/Electric-Vehicle-battery-failure-classification`.
4. Set Main file path: `app.py`.
5. Click **Deploy!**
