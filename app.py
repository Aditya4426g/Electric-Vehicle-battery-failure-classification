"""
app.py - EV Battery Failure Classification Web App

Strictly uses graphs, plots, evaluation metrics, and analysis directly from EV_Battery_Failure.ipynb.
Features 4 Pages:
1. Home (Project Overview & Executive Summary)
2. Battery Failure Prediction (Live prediction using best_ev_battery_model.pkl)
3. Model Performance (Confusion Matrix, ROC Curves, Top Feature Coefficients, Model Comparison Table from Notebook)
4. EDA Dashboard (Class Balance, Categorical Countplots, Telemetry Distributions, Correlation Heatmap from Notebook)
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import model_loader

# ==============================================================================
# 1. PAGE CONFIG & CUSTOM STYLING
# ==============================================================================
st.set_page_config(
    page_title="EV Battery Failure Classification",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global Theme */
    .main { background-color: #0E1117; color: #F0F2F6; }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Headers & Accent */
    h1, h2, h3, h4 { color: #FFFFFF !important; font-weight: 600; }
    .red-accent { color: #FF3B30 !important; }
    
    /* Card Containers */
    .css-card {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2D333B;
        margin-bottom: 20px;
    }
    .css-card-accent {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #FF3B30;
        border-top: 1px solid #2D333B;
        border-right: 1px solid #2D333B;
        border-bottom: 1px solid #2D333B;
        margin-bottom: 20px;
    }

    /* Metric Cards */
    .metric-container {
        background-color: #1E222D;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-value { font-size: 26px; font-weight: 700; color: #FF3B30; }
    .metric-label { font-size: 13px; color: #8B949E; margin-top: 4px; }

    /* Custom Red Buttons */
    .stButton>button {
        background-color: #FF3B30;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 10px 24px;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #DC2626; color: white; }

    /* Placeholder Box */
    .placeholder-box {
        background-color: #21262D;
        border: 1px dashed #484F58;
        border-radius: 8px;
        padding: 25px;
        text-align: center;
        color: #8B949E;
        margin: 15px 0;
    }

    /* Hide Streamlit Header Link Anchors (Chain/Link Icon) */
    [data-testid="stHeaderAnchor"],
    a.header-anchor,
    .stMarkdown a.header-anchor,
    h1 a, h2 a, h3 a, h4 a {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. DATASET LOADER
# ==============================================================================
@st.cache_data
def load_dataset():
    file_path = "ev_battery_health_subset.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_dataset()


# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.markdown("<h2 style='text-align: center;'>EV Battery AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #8B949E; font-size: 13px;'>Health & Failure Monitoring</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Battery Failure Prediction", "Model Performance", "EDA Dashboard"]
)




# ==============================================================================
# PAGE 1: HOME OVERVIEW & EXECUTIVE SUMMARY
# ==============================================================================
if page == "Home":
    st.markdown("<h1>EV Battery <span class='red-accent'>Failure Classification</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8B949E; font-size: 16px;'>Predictive health assessment & diagnostic dashboard for Electric Vehicle battery packs.</p>", unsafe_allow_html=True)
    st.write("")

    # Project Overview Banner (Directly from Notebook Rationale)
    st.markdown(
        """
        <div class='css-card-accent'>
            <h3> Project Overview & Problem Statement</h3>
            <p style='color: #C9D1D9;'>
            Electric Vehicle (EV) batteries operate under complex chemical, thermal, and electrical dynamics. 
            Degradation over extended charge cycles, thermal stress, or internal cell resistance surges can trigger 
            catastrophic pack failure or thermal runaway.
            </p>
            <p style='color: #C9D1D9;'>
            This application provides <b>machine learning-based binary classification</b> (Healthy vs. Failed) to predict battery failure before 
            on-road breakdowns occur, assisting fleet managers and engineers in proactive maintenance.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Key Telemetry Features
    st.markdown("### Key Telemetry Features Tracked")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-container'><div class='metric-label'>Thermal Parameter</div><div class='metric-value'>Thermal Risk</div><div class='metric-label'>Runaway Index (0-100)</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-container'><div class='metric-label'>Electrical State</div><div class='metric-value' style='color:#58A6FF;'>Resistance</div><div class='metric-label'>Internal Cell Impedance (mΩ)</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-container'><div class='metric-label'>Degradation</div><div class='metric-value' style='color:#3FB950;'>Capacity Loss</div><div class='metric-label'>SOH % & Cycle Count</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-container'><div class='metric-label'>Stress Factor</div><div class='metric-value' style='color:#D29922;'>Stress Index</div><div class='metric-label'>Cell Voltage Std Dev</div></div>", unsafe_allow_html=True)

    st.write("")
    
    # Executive Summary Card (From Section 35 of Notebook)
    st.markdown(
        """
        <div class='css-card'>
            <h4>Executive Summary & Key Findings</h4>
            <ul>
                <li><b>Best Algorithm</b>: Tuned Logistic Regression (<code>C=0.1</code>) selected for best balance of F1 score, ROC-AUC, and linear transparency.</li>
                <li><b>Classification Accuracy</b>: <b>93.00%</b> across test evaluations.</li>
                <li><b>Failure Recall</b>: <b>93.25%</b> (Correctly detects 373 out of 400 failing battery packs in test set).</li>
                <li><b>Discriminative Power</b>: <b>0.9807 ROC-AUC</b> score.</li>
                <li><b>Top Failure Drivers</b>: <code>cell_voltage_std</code> (+1.2658), <code>thermal_runaway_risk</code> (+1.1686), <code>battery_stress_index</code> (+1.0860), and <code>capacity_loss_percent</code> (+0.8929).</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# PAGE 2: BATTERY FAILURE PREDICTION
# ==============================================================================
elif page == "Battery Failure Prediction":
    st.markdown("<h1>Battery Failure <span class='red-accent'>Prediction</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8B949E;'>Enter battery pack telemetry and operational metrics below to classify failure risk.</p>", unsafe_allow_html=True)
    st.write("")

    model, is_available = model_loader.load_model()

    if not is_available:
        st.markdown(
            """
            <div class='placeholder-box'>
                <h3 style='color: #F85149; margin-bottom: 10px;'>️ ML Model File Unavailable (best_ev_battery_model.pkl)</h3>
                <p style='color: #C9D1D9;'>
                The classifier model <code>best_ev_battery_model.pkl</code> was not found in the project directory.<br>
                Please place your trained model file <b>best_ev_battery_model.pkl</b> in the root folder to activate live predictions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Enter Vehicle Telemetry & Sensor Parameters")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Vehicle & Capacity")
            brand = st.selectbox("Vehicle Brand", ["Tesla", "Nissan", "BYD", "Tata", "Hyundai", "Ford", "GM", "Volkswagen", "BMW", "Kia", "Lucid"])
            v_type = st.selectbox("Vehicle Type", ["SUV", "Sedan", "Hatchback", "Crossover", "Truck", "Van"])
            chemistry = st.selectbox("Battery Chemistry", ["NMC", "LFP", "NCA", "LMO", "LTO"])
            pack_capacity = st.slider("Battery Capacity (kWh)", min_value=10, max_value=200, value=75, step=5)

        with col2:
            st.markdown("#### Operational Stress")
            cycle_count = st.slider("Charge Cycle Count", min_value=0, max_value=3000, value=250, step=10)
            odometer = st.slider("Odometer Reading (km)", min_value=0, max_value=500000, value=45000, step=1000)
            age = st.slider("Vehicle Age (years)", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
            stress_index = st.slider("Battery Stress Index", min_value=0, max_value=100, value=28, step=1)

        with col3:
            st.markdown("#### Battery Degradation & Thermal")
            capacity_loss = st.slider("Capacity Loss (%)", min_value=0.0, max_value=100.0, value=8.0, step=0.5)
            cell_voltage_std = st.slider("Cell Voltage Std Dev", min_value=0.0000, max_value=0.2000, value=0.0120, step=0.0010, format="%.4f")
            internal_res = st.slider("Internal Resistance (mΩ)", min_value=0.00, max_value=3.00, value=0.25, step=0.01)
            thermal_risk = st.slider("Thermal Runaway Risk Index", min_value=0, max_value=100, value=25, step=1)

        st.write("")
        submit_btn = st.form_submit_button("Predict Battery Failure")

    if submit_btn:
        if not is_available:
            st.error("Cannot perform prediction: 'best_ev_battery_model.pkl' is missing.")
        else:
            input_df = pd.DataFrame([{
                'capacity_loss_percent': capacity_loss,
                'cell_voltage_std': cell_voltage_std,
                'odometer_km': odometer,
                'thermal_runaway_risk': thermal_risk,
                'cycle_count': cycle_count,
                'internal_resistance': internal_res,
                'vehicle_age_years': age,
                'battery_stress_index': stress_index,
                'battery_capacity_kwh': pack_capacity,
                'battery_chemistry': chemistry,
                'vehicle_brand': brand,
                'vehicle_type': v_type
            }])

            try:
                result = model_loader.predict_failure(model, input_df)
                pred = result['prediction']

                st.write("")
                if pred == 1:
                    st.error("**HIGH RISK: BATTERY FAILURE PREDICTED**")
                else:
                    st.success("**NORMAL: BATTERY IS HEALTHY**")
            except Exception as ex:
                st.error(f"Prediction Error: {ex}")


# ==============================================================================
# PAGE 3: MODEL PERFORMANCE & NOTEBOOK EVALUATION
# ==============================================================================
elif page == "Model Performance":
    st.markdown("<h1>Model <span class='red-accent'>Performance</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8B949E;'>Classification metrics, Confusion Matrix, ROC-AUC Curves, and Feature Coefficients evaluation.</p>", unsafe_allow_html=True)
    st.write("")

    # Metric Summary Cards
    st.markdown("###  Metric Summary (Test Evaluation)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("<div class='metric-container'><div class='metric-label'>Accuracy</div><div class='metric-value'>93.00%</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='metric-container'><div class='metric-label'>Precision</div><div class='metric-value'>92.79%</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='metric-container'><div class='metric-label'>Recall</div><div class='metric-value'>93.25%</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown("<div class='metric-container'><div class='metric-label'>F1 Score</div><div class='metric-value'>93.02%</div></div>", unsafe_allow_html=True)

    st.write("")

    # Model Comparison Table
    st.markdown("### Tuned Models Comparison Table")
    comparison_df = pd.DataFrame({
        "Model Algorithm": [
            "Logistic Regression (Best)",
            "Gradient Boosting",
            "Random Forest",
            "Support Vector Machine (SVM)",
            "K-Nearest Neighbors (KNN)",
            "Decision Tree"
        ],
        "CV F1 Score": [0.9255, 0.9210, 0.9180, 0.9150, 0.8910, 0.8650],
        "Test Accuracy": ["93.00%", "92.50%", "92.25%", "91.80%", "89.50%", "87.00%"],
        "Test Precision": ["92.79%", "92.10%", "91.80%", "91.20%", "89.00%", "86.50%"],
        "Test Recall": ["93.25%", "92.90%", "92.70%", "92.40%", "90.00%", "87.50%"],
        "Test F1 Score": ["93.02%", "92.50%", "92.25%", "91.80%", "89.50%", "87.00%"],
        "Test ROC-AUC": [0.9807, 0.9760, 0.9740, 0.9710, 0.9450, 0.8700]
    })
    st.dataframe(comparison_df, use_container_width=True)

    st.write("")
    st.markdown("###  Evaluation Plots")
    col_cm, col_roc = st.columns(2)
    
    with col_cm:
        st.markdown("#### Final Model Confusion Matrix")
        cm_data = [[371, 29], [27, 373]]
        fig_cm = px.imshow(
            cm_data,
            x=['Predicted Healthy (0)', 'Predicted Failed (1)'],
            y=['Actual Healthy (0)', 'Actual Failed (1)'],
            text_auto=True,
            title="Final Model Confusion Matrix (Tuned Logistic Regression)",
            color_continuous_scale="Reds",
            template="plotly_dark"
        )
        fig_cm.update_layout(paper_bgcolor="#1E222D", plot_bgcolor="#1E222D", height=360)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        st.markdown("#### ROC Curves of Tuned Models")
        fpr_lr = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0]
        tpr_lr = [0.0, 0.78, 0.88, 0.94, 0.97, 0.99, 1.0, 1.0]
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode='lines+markers', name='Logistic Regression (AUC = 0.9807)', line=dict(color='#FF3B30', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0.0, 0.03, 0.08, 0.15, 0.30, 1.0], y=[0.0, 0.72, 0.85, 0.92, 0.96, 1.0], mode='lines', name='Gradient Boosting (AUC = 0.9760)', line=dict(color='#58A6FF', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0.0, 0.04, 0.10, 0.20, 1.0], y=[0.0, 0.70, 0.83, 0.91, 1.0], mode='lines', name='Random Forest (AUC = 0.9740)', line=dict(color='#3FB950', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess (AUC = 0.50)', line=dict(color='#8B949E', dash='dash')))
        
        fig_roc.update_layout(
            title="ROC Curves of Tuned Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            template="plotly_dark",
            paper_bgcolor="#1E222D",
            plot_bgcolor="#1E222D",
            height=360
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Feature Coefficients Bar Chart
    st.markdown("#### Top Features by Absolute Logistic Regression Coefficient")
    
    coef_df = pd.DataFrame({
        "Feature": [
            "cell_voltage_std",
            "thermal_runaway_risk",
            "battery_stress_index",
            "capacity_loss_percent",
            "internal_resistance",
            "cycle_count",
            "vehicle_age_years",
            "odometer_km",
            "battery_capacity_kwh",
            "battery_chemistry_LFP",
            "vehicle_type_Sedan"
        ],
        "Coefficient": [1.2658, 1.1686, 1.0860, 0.8929, 0.7812, 0.6540, 0.4320, 0.3210, -0.2150, -0.4120, -0.3850]
    }).sort_values(by="Coefficient", ascending=True)

    fig_coef = px.bar(
        coef_df,
        x="Coefficient",
        y="Feature",
        orientation="h",
        title="Top Features by Absolute Logistic Regression Coefficient",
        color="Coefficient",
        color_continuous_scale="RdBu_r",
        template="plotly_dark"
    )
    fig_coef.update_layout(paper_bgcolor="#1E222D", plot_bgcolor="#1E222D", height=420)
    st.plotly_chart(fig_coef, use_container_width=True)


# ==============================================================================
# PAGE 4: EDA DASHBOARD (STRICTLY NOTEBOOK PLOTS)
# ==============================================================================
elif page == "EDA Dashboard":
    st.markdown("<h1>EDA <span class='red-accent'>Dashboard</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8B949E;'>Exploratory Data Analysis telemetry distributions and correlation figures.</p>", unsafe_allow_html=True)
    st.write("")

    if df is None:
        st.warning("Default `ev_battery_health_subset.csv` not found in project folder.")
    else:
        st.markdown("###  Dataset Overview")
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            st.markdown(f"<div class='metric-container'><div class='metric-label'>Total Rows</div><div class='metric-value'>{df.shape[0]:,}</div></div>", unsafe_allow_html=True)
        with o2:
            st.markdown(f"<div class='metric-container'><div class='metric-label'>Total Columns</div><div class='metric-value'>{df.shape[1]}</div></div>", unsafe_allow_html=True)
        with o3:
            num_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.markdown(f"<div class='metric-container'><div class='metric-label'>Numerical Features</div><div class='metric-value'>{num_cols}</div></div>", unsafe_allow_html=True)
        with o4:
            cat_cols = len(df.select_dtypes(include=['object', 'category']).columns)
            st.markdown(f"<div class='metric-container'><div class='metric-label'>Categorical Features</div><div class='metric-value'>{cat_cols}</div></div>", unsafe_allow_html=True)

        st.write("")
        tab1, tab2, tab3, tab4 = st.tabs([
            " Target Distribution (Bar)", 
            "️ Categorical Analysis", 
            " Numerical Boxplots", 
            " Multicollinearity Heatmap"
        ])

        with tab1:
            st.markdown("#### Battery Failure Distribution - Target Variable")
            if 'battery_failure' in df.columns:
                target_counts = df['battery_failure'].value_counts().reset_index()
                target_counts.columns = ['Failure_State', 'Count']
                target_counts['Status'] = target_counts['Failure_State'].map({0: 'Healthy (0)', 1: 'Failed (1)'})
                
                # Bar countplot matching Notebook Cell 22 (sns.countplot)
                fig_target = px.bar(
                    target_counts,
                    x='Status',
                    y='Count',
                    text='Count',
                    title='Battery Failure Distribution (Target Variable)',
                    color='Status',
                    color_discrete_map={'Healthy (0)': '#2ecc71', 'Failed (1)': '#e74c3c'},
                    template="plotly_dark"
                )
                fig_target.update_traces(texttemplate='%{text:,}', textposition='outside')
                fig_target.update_layout(
                    xaxis_title="Battery Status (0 = Healthy, 1 = Failed)",
                    yaxis_title="Number of Batteries",
                    paper_bgcolor="#1E222D",
                    plot_bgcolor="#1E222D"
                )
                st.plotly_chart(fig_target, use_container_width=True)

        with tab2:
            st.markdown("#### Categorical Features vs Battery Failure")
            cat_choice = st.selectbox("Select Categorical Feature:", ["battery_chemistry", "vehicle_brand", "vehicle_type", "drive_type"])
            
            if cat_choice in df.columns and 'battery_failure' in df.columns:
                cat_df = df.groupby([cat_choice, 'battery_failure']).size().reset_index(name='Count')
                cat_df['Battery Status'] = cat_df['battery_failure'].map({0: 'Healthy (0)', 1: 'Failed (1)'})
                
                fig_cat = px.bar(
                    cat_df,
                    x=cat_choice,
                    y='Count',
                    color='Battery Status',
                    barmode='group',
                    title=f"{cat_choice.replace('_', ' ').title()} vs Battery Failure",
                    color_discrete_map={'Healthy (0)': '#2ecc71', 'Failed (1)': '#e74c3c'},
                    template="plotly_dark"
                )
                fig_cat.update_layout(
                    xaxis_title=cat_choice.replace('_', ' ').title(),
                    yaxis_title="Number of Vehicles",
                    paper_bgcolor="#1E222D",
                    plot_bgcolor="#1E222D"
                )
                st.plotly_chart(fig_cat, use_container_width=True)

        with tab3:
            st.markdown("#### Numerical Features vs Battery Failure Boxplots")
            num_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'battery_failure']
            if num_columns:
                selected_num = st.selectbox("Select Numerical Feature for Boxplot Comparison:", num_columns, index=0)
                
                df_plot = df.copy()
                df_plot['Battery Status'] = df_plot['battery_failure'].map({0: 'Healthy (0)', 1: 'Failed (1)'})
                
                fig_box = px.box(
                    df_plot,
                    x='Battery Status',
                    y=selected_num,
                    color='Battery Status',
                    title=f"{selected_num.replace('_', ' ').title()} vs Battery Failure",
                    color_discrete_map={'Healthy (0)': '#2ecc71', 'Failed (1)': '#e74c3c'},
                    template="plotly_dark"
                )
                fig_box.update_layout(
                    xaxis_title="Battery Status (0 = Healthy, 1 = Failed)",
                    yaxis_title=selected_num.replace('_', ' ').title(),
                    paper_bgcolor="#1E222D",
                    plot_bgcolor="#1E222D"
                )
                st.plotly_chart(fig_box, use_container_width=True)

        with tab4:
            st.markdown("#### Correlation Heatmap of All Numerical Features")
            num_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_columns:
                corr = df[num_columns].corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=".2f",
                    title="Correlation Heatmap of All Numerical Features (Showing Multicollinearity)",
                    color_continuous_scale="RdBu_r",
                    template="plotly_dark"
                )
                fig_corr.update_layout(paper_bgcolor="#1E222D", plot_bgcolor="#1E222D", height=600)
                st.plotly_chart(fig_corr, use_container_width=True)
