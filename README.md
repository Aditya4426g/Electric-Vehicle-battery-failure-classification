# EV Battery Failure Classification & Health Monitoring Web Application

Predictive analytics and failure classification dashboard for Electric Vehicle (EV) battery packs using Machine Learning.

🚀 **Live Web Application**: [https://electric-vehicle-battery-failure-classification.streamlit.app](https://electric-vehicle-battery-failure-classification.streamlit.app)

---

## 📌 Overview

This application provides real-time binary classification (**Healthy** vs. **High Risk Failure**) to monitor EV battery degradation based on vehicle telemetry and operational stress parameters. Built using Streamlit, Python, Scikit-Learn, and Plotly.

---

## 📁 Directory Structure

```text
Electric-Vehicle-battery-failure-classification/
├── data/
│   └── ev_battery_health_subset.csv       # Telemetry dataset (4,000 records)
├── models/
│   └── best_ev_battery_model.pkl          # Single bundled ML model artifact (Model + Scaler + Encodings)
├── docs/
│   └── EV_Battery_Failure.ipynb           # Model development & EDA notebook
├── app.py                                 # Streamlit Web Application entrypoint
├── model_loader.py                        # Single-pkl preprocessing & inference pipeline
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
└── README.md                              # Project documentation
```

---

## 📊 Machine Learning Performance

* **Best Model Algorithm**: Tuned Logistic Regression (`C=0.01`)
* **Test Classification Accuracy**: **92.00%**
* **Failure Recall (Detection Rate)**: **93.00%** (Correctly detects 372 out of 400 failing battery packs)
* **Discriminative Power**: **0.9789 ROC-AUC**
* **Test F1 Score**: **92.08%**
* **Model Artifact**: Single self-contained pickle bundle (`best_ev_battery_model.pkl`)

---

## ⚡ Key Telemetry Features (12 Features)

### 9 Numerical Features:
1. `capacity_loss_percent` — State of Health degradation (%)
2. `cell_voltage_std` — Cell voltage imbalance (V)
3. `odometer_km` — Total cumulative mileage (km)
4. `thermal_runaway_risk` — Thermal runaway risk score (0–100)
5. `cycle_count` — Full charge/discharge cycles
6. `internal_resistance` — Cell impedance (mΩ)
7. `vehicle_age_years` — Vehicle operational age (years)
8. `battery_stress_index` — Composite stress score (0–100)
9. `battery_capacity_kwh` — Pack energy capacity (kWh)

### 3 Categorical Features:
1. `battery_chemistry` — Cell chemistry (NMC, LFP, NCA, LMO, LTO)
2. `vehicle_brand` — Automotive OEM brand
3. `vehicle_type` — Form factor (SUV, Sedan, Hatchback, Crossover, Truck, Van)

---

## 💻 Web Application Pages

1. **Home**: Project summary, key findings, and core telemetry metrics.
2. **Battery Failure Prediction**: Real-time interactive telemetry risk classification powered by `best_ev_battery_model.pkl`.
3. **Model Performance**: Accuracy metrics summary, Tuned Model Comparison Table & Bar Chart, Confusion Matrix, and ROC Curves.
4. **EDA Dashboard**: Target Variable Distribution, Categorical Feature Plots, Telemetry Histograms, and 12-Feature Correlation Heatmap.

---

## 🛠️ Local Installation & Quickstart

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

## ☁️ Live Cloud Deployment

Access the deployed application live at:
👉 **[https://electric-vehicle-battery-failure-classification.streamlit.app](https://electric-vehicle-battery-failure-classification.streamlit.app)**
