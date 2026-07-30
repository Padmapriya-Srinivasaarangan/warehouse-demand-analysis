# ======================================================
# 🔮 Time Series Forecasting Dashboard (with Residual Plots)
# ======================================================

import pandas as pd
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt



st.set_page_config(page_title="Time Series Forecast", page_icon="🔮")
st.title("🔮 Time Series Forecasting")

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




# -----------------------------------------------------
# Load Data
# -----------------------------------------------------
df = pd.read_csv("cleaned_warehouse_data.csv")

ts_df = df[['date', 'demand']].copy()
ts_df['date'] = pd.to_datetime(ts_df['date'], errors='coerce')
ts_df = ts_df.groupby('date')['demand'].sum().reset_index()
ts_df.columns = ['ds', 'y']

# -----------------------------------------------------
# Fit Prophet Model
# -----------------------------------------------------
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.fit(ts_df)

# -----------------------------------------------------
# Forecast (Next 30 Days)
# -----------------------------------------------------
st.subheader("📆 Daily Forecast (Next 30 Days)")

future_daily = model.make_future_dataframe(periods=30)
forecast_daily = model.predict(future_daily)

fig1 = model.plot(forecast_daily)
st.pyplot(fig1)

st.info(
    "📝 **Explanation:** This chart predicts the expected demand for the next 30 days. "
    "The blue line shows the predicted values and the shaded region shows the uncertainty range."
)

# -----------------------------------------------------
# Weekly Forecast (Next 12 Weeks)
# -----------------------------------------------------
st.subheader("📆 Weekly Forecast (Next 12 Weeks)")

forecast_daily.set_index("ds", inplace=True)
weekly_pred = forecast_daily["yhat"].resample("W").sum().reset_index()

fig2, ax = plt.subplots(figsize=(10, 4))
ax.plot(weekly_pred["ds"], weekly_pred["yhat"], marker="o")
ax.set_title("Weekly Forecast")
ax.set_xlabel("Week")
ax.set_ylabel("Predicted Weekly Demand")
ax.grid(True)
st.pyplot(fig2)

st.info(
    "📝 **Explanation:** Weekly forecasting helps in planning procurement cycles and shipment scheduling."
)

# ======================================================
# 📉 Residual Analysis Section
# ======================================================
st.markdown("---")
st.subheader("📉 Residual Analysis")

# Prophet provides predictions for historical dates → extract those
historical = forecast_daily.reset_index().merge(
    ts_df, on="ds", how="inner"
)

y_true = historical["y"]
y_pred = historical["yhat"]
residuals = y_true - y_pred

# -----------------------------------------------------
# Plot 1: Actual vs Predicted
# -----------------------------------------------------
st.markdown("### 🎯 Actual vs Predicted Values")

fig3, ax3 = plt.subplots()
ax3.scatter(y_true, y_pred)
ax3.set_xlabel("Actual Demand")
ax3.set_ylabel("Predicted Demand")
ax3.set_title("Actual vs Predicted")
st.pyplot(fig3)

st.info("""
**Explanation:**  
This plot compares the true historical demand with the model’s predicted values.  
- Points close to a diagonal line indicate good accuracy.  
- Large deviations highlight periods where the model struggled to follow actual demand.  
Useful for visually checking overall forecasting performance.
""")


# -----------------------------------------------------
# Plot 2: Residuals vs Actual
# -----------------------------------------------------
st.markdown("### 📌 Residuals vs Actual Demand")

fig4, ax4 = plt.subplots()
ax4.scatter(y_true, residuals)
ax4.axhline(0, color='black')
ax4.set_xlabel("Actual Demand")
ax4.set_ylabel("Residuals")
ax4.set_title("Residuals vs Actual")
st.pyplot(fig4)

st.info("""
**Explanation:**  
Residuals represent the forecasting error:  
**Residual = Actual – Predicted**  
This plot helps you see:  
- Whether errors grow with increasing demand  
- Whether the model underestimates high-demand days or low-demand days  
A good model shows random scattering around zero without patterns.
""")


# -----------------------------------------------------
# Plot 3: Residuals vs Predicted
# -----------------------------------------------------
st.markdown("### 📌 Residuals vs Predicted Values")

fig5, ax5 = plt.subplots()
ax5.scatter(y_pred, residuals)
ax5.axhline(0, color='black')
ax5.set_xlabel("Predicted Demand")
ax5.set_ylabel("Residuals")
ax5.set_title("Residuals vs Predicted")
st.pyplot(fig5)

st.info("""
**Explanation:**  
This chart checks whether the model is consistently biased:  
- If residuals shift above 0 → underestimation  
- If residuals shift below 0 → overestimation  
- A good model shows no pattern, meaning predictions are balanced.
""")


# -----------------------------------------------------
# Plot 4: Residual Histogram
# -----------------------------------------------------
st.markdown("### 📊 Residual Distribution (Histogram)")

fig6, ax6 = plt.subplots()
ax6.hist(residuals, bins=30)
ax6.set_xlabel("Residual Value")
ax6.set_ylabel("Frequency")
ax6.set_title("Residual Distribution")
st.pyplot(fig6)

st.info("""
**Explanation:**  
The histogram shows how often particular error values occur.  
A well-performing model has:  
- A peak near zero (small errors)  
- A symmetric, bell-shaped distribution  
If the distribution is skewed or widely spread, the model may require tuning.
""")

st.success("Residual analysis completed successfully.")
