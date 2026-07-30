# ======================================================
# 🤖 Random Forest Model Dashboard
# ======================================================

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np


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

# -----------------------------
# Page Title
# -----------------------------
st.title("🤖 Random Forest Model")

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("cleaned_warehouse_data.csv")

# -----------------------------
# Feature Engineering
# -----------------------------
df['stock_to_cost_ratio'] = df['on_hand'] / (df['unit_cost'] + 1)
df['lead_time_efficiency'] = df['on_hand'] / (df['lead_time_days'] + 1)
df['cost_efficiency'] = df['demand'] / (df['unit_cost'] + 1)

# -----------------------------
# Features & Target
# -----------------------------
features = ['on_hand', 'lead_time_days', 'unit_cost',
            'stock_to_cost_ratio', 'lead_time_efficiency', 'cost_efficiency']
target = 'demand'

X = df[features]
y = df[target]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Random Forest
# -----------------------------
model = RandomForestRegressor(n_estimators=200, max_features='sqrt', random_state=42)
model.fit(X_train, y_train)

# Save Model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/random_forest.pkl")

# -----------------------------
# Predictions & Metrics
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

# -----------------------------
# Metrics Display
# -----------------------------
st.subheader("📊 Evaluation Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.3f}")
col4.metric("MAPE", f"{mape:.2f}%")

st.info("""
📝 **Explanation:**
- MAE and RMSE measure the average prediction error size.
- R² Score indicates how well the model fits the data.
- **MAPE shows prediction error in percentage terms**, making it easy to understand model accuracy.
- Lower MAPE means better prediction accuracy for warehouse demand forecasting.
""")

# -----------------------------
# Actual vs Predicted Scatter Plot
# -----------------------------
st.subheader("🎯 Actual vs Predicted Demand")
fig, ax = plt.subplots(figsize=(7,4))
sns.scatterplot(x=y_test, y=y_pred, color='purple', alpha=0.6, ax=ax)
ax.set_xlabel("Actual Demand")
ax.set_ylabel("Predicted Demand")
ax.set_title("Random Forest: Actual vs Predicted")
st.pyplot(fig)

st.info("""
📝 **Explanation:**
- Each point compares real demand vs. predicted demand.
- A perfect model would produce points that lie on a straight diagonal line.
- More clustering around the line indicates higher prediction accuracy.
""")

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("📌 Feature Importance")
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
fig2, ax2 = plt.subplots(figsize=(6,4))
importance.plot(kind='barh', color='teal', ax=ax2)
ax2.set_xlabel("Importance Score")
ax2.set_title("Feature Importance (Random Forest)")
st.pyplot(fig2)

st.info("""
📝 **Explanation:**
- Features with higher scores have a stronger influence on demand prediction.
- Helps identify operational drivers affecting warehouse demand.
- Useful for business decisions such as inventory planning and cost control.
""")