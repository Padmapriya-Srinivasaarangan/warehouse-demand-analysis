# ======================================================
# ⚡ Demand Prediction Dashboard / Prediction Tool
# ======================================================

import streamlit as st
import pandas as pd
import joblib
from prophet import Prophet
import matplotlib.pyplot as plt


st.markdown(
    """
    <style>
    /* ===== GLOBAL APP ===== */
    .stApp {
        background-color: #0A0F1C;
        color: #EDEFF2;
        font-family: "Segoe UI", sans-serif;
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background-color: #0A0F1C;
        border-right: 1px solid #1E293B;
        padding: 20px;
    }

    section[data-testid="stSidebar"] * {
        color: #EDEFF2 !important;
        font-weight: 600;
    }

    /* Sidebar radio buttons fix */
    section[data-testid="stSidebar"] label span {
        color: #EDEFF2 !important;
    }

    /* ===== HEADINGS ===== */
    h1, h2, h3, h4, h5 {
        color: #FFFFFF;
        font-weight: 700;
    }

    /* ===== NORMAL TEXT ===== */
    p, span, label, div {
        color: #E5E7EB;
    }

    /* ===== METRIC CARDS ===== */
    div[data-testid="stMetric"] {
        background-color: #111827;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #1E293B;
        box-shadow: 0 12px 32px rgba(0,0,0,0.55);
    }

    div[data-testid="stMetricLabel"] {
        color: #9CA3AF !important;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 26px;
        font-weight: 700;
    }

    /* ===== BUTTONS ===== */
    button {
        background-color: #111827 !important;
        color: #FFFFFF !important;
        border: 1px solid #1E293B;
        border-radius: 10px;
        font-weight: 600;
    }

    button:hover {
        background-color: #1E293B !important;
    }

    /* ===== DATAFRAME ===== */
    .stDataFrame {
        background-color: #111827 !important;
        border-radius: 12px;
        border: 1px solid #1E293B;
        padding: 10px;
        font-size: 14px;
        color: #EDEFF2 !important;
    }

    .stDataFrame thead th {
        background-color: #0A0F1C !important;
        color: #FFFFFF !important;
        font-weight: 700;
    }

    .stDataFrame tbody td {
        color: #EDEFF2 !important;
    }

    .stDataFrame tbody tr:hover {
        background-color: #1E293B;
    }

    /* ===== TABS ===== */
    button[data-baseweb="tab"] {
        background-color: #0A0F1C !important;
        color: #9CA3AF !important;
        border-bottom: 2px solid transparent;
        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("⚡ Demand Prediction Dashboard")



# -----------------------------
# 1️⃣ RANDOM FOREST PREDICTION
# -----------------------------
st.header("🎯 Random Forest Daily Demand Prediction")

# Load the trained RF model
rf_model = joblib.load("models/random_forest.pkl")

st.subheader("Enter Feature Values:")
on_hand = st.number_input("On-Hand Stock", min_value=0)
lead_time = st.number_input("Lead Time (days)", min_value=0)
unit_cost = st.number_input("Unit Cost", min_value=0.0, format="%.2f")

if st.button("Predict Daily Demand"):
    # Feature Engineering (same as training stage)
    stock_to_cost_ratio = on_hand / (unit_cost + 1)
    lead_time_efficiency = on_hand / (lead_time + 1)
    cost_efficiency = 0  # default placeholder

    input_df = pd.DataFrame([[on_hand, lead_time, unit_cost,
                              stock_to_cost_ratio, lead_time_efficiency, cost_efficiency]],
                            columns=['on_hand','lead_time_days','unit_cost',
                                     'stock_to_cost_ratio','lead_time_efficiency','cost_efficiency'])
    
    pred = rf_model.predict(input_df)[0]
    st.success(f"✅ Predicted Daily Demand: {pred:.2f} units")

