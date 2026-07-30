import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# ======================================================
# Page Config (MUST be first Streamlit command)
# ======================================================
st.set_page_config(
    page_title="Export Reports",
    page_icon="📁",
    layout="wide"
)

# ======================================================
# Custom Styling
# ======================================================
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

# ======================================================
# Title
# ======================================================
st.title("📁 Export Reports & Artifacts")

# ======================================================
# Load Cleaned Dataset
# ======================================================
if os.path.exists("cleaned_warehouse_data.csv"):
    df = pd.read_csv("cleaned_warehouse_data.csv")
else:
    st.error("❌ cleaned_warehouse_data.csv not found.")
    st.stop()

# ======================================================
# Export Raw Dataset
# ======================================================
st.subheader("📄 Download Original Dataset")
raw_csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download Dataset (CSV)",
    data=raw_csv,
    file_name="warehouse_cleaned_data.csv",
    mime="text/csv"
)

st.markdown("---")

# ======================================================
# Load Trained Model
# ======================================================
model_path = "models/random_forest.pkl"

if not os.path.exists(model_path):
    st.error("❌ Random Forest model not found.")
    st.stop()

model = joblib.load(model_path)

# ======================================================
# Feature Engineering (MUST match training)
# ======================================================
try:
    df["stock_to_cost_ratio"] = df["on_hand"] / (df["unit_cost"] + 1)
    df["lead_time_efficiency"] = df["on_hand"] / (df["lead_time_days"] + 1)
    df["cost_efficiency"] = df["demand"] / (df["unit_cost"] + 1)
except KeyError as e:
    st.error(f"❌ Missing column: {e}")
    st.stop()

# ======================================================
# Prediction
# ======================================================
features = [
    "on_hand",
    "lead_time_days",
    "unit_cost",
    "stock_to_cost_ratio",
    "lead_time_efficiency",
    "cost_efficiency"
]

X = df[features].fillna(0)
y_pred = model.predict(X)

df_pred = df.copy()
df_pred["predicted_demand"] = y_pred

# ======================================================
# Download Predictions
# ======================================================
st.subheader("📦 Download Predictions")

st.download_button(
    "⬇️ Download Predictions (CSV)",
    data=df_pred.to_csv(index=False).encode("utf-8"),
    file_name="warehouse_predictions.csv",
    mime="text/csv"
)

# ======================================================
# Export Metrics
# ======================================================
if "demand" in df.columns:
    mae = mean_absolute_error(df["demand"], y_pred)
    mse = mean_squared_error(df["demand"], y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(df["demand"], y_pred)
    mape = mean_absolute_percentage_error(df["demand"], y_pred) * 100

    metrics_text = f"""
Random Forest Model Metrics
---------------------------
MAE  : {mae:.3f}
RMSE : {rmse:.3f}
R²   : {r2:.3f}
MAPE : {mape:.2f}%
"""

    st.subheader("📊 Download Metrics Summary")
    st.download_button(
        "⬇️ Download Metrics Report (TXT)",
        data=metrics_text,
        file_name="random_forest_metrics.txt",
        mime="text/plain"
    )
else:
    st.warning("⚠️ Demand column missing. Metrics unavailable.")
