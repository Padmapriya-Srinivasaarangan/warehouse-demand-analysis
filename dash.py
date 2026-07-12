# ======================================================
# 📦 Warehouse Dashboard - All-in-One (Streamlit)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
px.defaults.template = "plotly_dark"

from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score





# ======================================================
# 0️⃣ Streamlit Page Config
# ======================================================
st.set_page_config(page_title="Warehouse Dashboard", page_icon="📦", layout="wide")
plt.style.use("dark_background")
sns.set_theme(style="dark")


st.sidebar.title("Filters & Pages")

# Page selection
page_option = st.sidebar.radio(
    "Select Dashboard Page", 
    ["Overview", "Visualizations", "ML Model", "Time Series Forecast"]
)


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



# 2️⃣ Load & Clean Data
# ======================================================
@st.cache_data
def load_and_clean_data(file_name="warehouse_data_100000.csv"):
    df = pd.read_csv(file_name)

    # Numeric and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Fill missing
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove duplicates
    df = df.drop_duplicates()

    # Convert numeric safely
    numeric_features = ['demand', 'on_hand', 'lead_time_days', 'unit_cost']
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove negatives
    for col in numeric_features:
        if col in df.columns:
            df = df[df[col] >= 0]

    # Remove zero demand
    if 'demand' in df.columns:
        df = df[df['demand'] > 0]

    # Convert 'date' to datetime safely
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'].notna()].copy()
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year

    # Normalize text (exclude date column)
    for col in cat_cols:
        if col != 'date':
            df[col] = df[col].astype(str).str.strip().str.lower()


    # Feature engineering
    if {'on_hand','unit_cost'}.issubset(df.columns):
        df['inventory_value'] = df['on_hand'] * df['unit_cost']

    return df

df = load_and_clean_data()

# Ensure date column is datetime after all processing
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')



# ======================================================
# 🔎 Sidebar Filters (Robust Version)
# ======================================================

st.sidebar.title("Filters & Pages")
st.sidebar.subheader("Advanced Filters")

# 1️⃣ Date Filter
if 'date' in df.columns:
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()

    date_input = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Ensure unpacking works
    if isinstance(date_input, (tuple, list)):
        start_date, end_date = date_input
    else:
        start_date = end_date = date_input

    df = df[(df['date'] >= pd.to_datetime(start_date)) &
            (df['date'] <= pd.to_datetime(end_date))]

# 2️⃣ Demand Filter
if 'demand' in df.columns:
    min_demand = int(df['demand'].min())
    max_demand = int(df['demand'].max())

    demand_range = st.sidebar.slider(
        "Demand Range",
        min_value=min_demand,
        max_value=max_demand,
        value=(min_demand, max_demand)
    )
    df = df[(df['demand'] >= demand_range[0]) &
            (df['demand'] <= demand_range[1])]

# 3️⃣ Unit Cost Filter
if 'unit_cost' in df.columns:
    min_cost = float(df['unit_cost'].min())
    max_cost = float(df['unit_cost'].max())

    cost_range = st.sidebar.slider(
        "Unit Cost Range",
        min_value=min_cost,
        max_value=max_cost,
        value=(min_cost, max_cost)
    )
    df = df[(df['unit_cost'] >= cost_range[0]) &
            (df['unit_cost'] <= cost_range[1])]

# 4️⃣ Stock Level Filter
if 'on_hand' in df.columns:
    stock_level = st.sidebar.selectbox(
        "Stock Level",
        ["All", "Low Stock", "Medium Stock", "High Stock"]
    )

    if stock_level == "Low Stock":
        df = df[df['on_hand'] < 50]
    elif stock_level == "Medium Stock":
        df = df[(df['on_hand'] >= 50) & (df['on_hand'] < 200)]
    elif stock_level == "High Stock":
        df = df[df['on_hand'] >= 200]

# 5️⃣ Category Filter
if 'category' in df.columns:
    category_options = df['category'].unique().tolist()
    category_filter = st.sidebar.multiselect(
        "Filter by Category",
        options=category_options,
        default=category_options
    )
    if category_filter:
        df = df[df['category'].isin(category_filter)]

# 6️⃣ Check if data is available
if df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()


# ======================================================
# 3️⃣ Overview Page
# ======================================================
if page_option == "Overview":
    st.title("📊 Warehouse Dashboard Overview")

    # KPIs
    total_demand = df['demand'].sum() if 'demand' in df.columns else 0
    avg_demand = df['demand'].mean() if 'demand' in df.columns else 0
    total_stock = df['on_hand'].sum() if 'on_hand' in df.columns else 0
    total_inventory_value = df['inventory_value'].sum() if 'inventory_value' in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Demand", f"{total_demand:.0f}")
    col2.metric("Average Demand", f"{avg_demand:.2f}")
    col3.metric("Total Stock On-Hand", f"{total_stock:.0f}")
    col4.metric("Total Inventory Value", f"${total_inventory_value:.2f}")

    # ===== ADVANCED KPIs =====
    avg_inventory = df['on_hand'].mean()
    inventory_turnover = total_demand / avg_inventory if avg_inventory > 0 else 0

    stockout_rate = (df[df['on_hand'] == 0].shape[0] / df.shape[0]) * 100

    overstock_threshold = 200
    overstock_cost = (
        (df[df['on_hand'] > overstock_threshold]['on_hand'] - overstock_threshold) *
        df[df['on_hand'] > overstock_threshold]['unit_cost']
    ).sum()

    avg_lead_time = df['lead_time_days'].mean()

    st.markdown("### 📌 Inventory Performance KPIs")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Inventory Turnover Ratio", f"{inventory_turnover:.2f}")
    col6.metric("Stockout Rate", f"{stockout_rate:.1f}%")
    col7.metric("Overstock Cost", f"${overstock_cost:,.0f}")
    col8.metric("Avg Lead Time", f"{avg_lead_time:.1f} days")

    st.subheader("Sample Data")
    st.dataframe(df.head(10))

    # ======================================================
    # 🔔 Stock Level Alerts Section
    # ======================================================
    st.subheader("🔔 Stock Level Alerts")

    # Define thresholds
    low_threshold = 50
    medium_threshold = 200

    if 'on_hand' in df.columns:
        avg_stock = df['on_hand'].mean()
        if avg_stock < low_threshold:
            st.warning(f"⚠️ Average stock is low: {avg_stock:.0f} units")
        elif avg_stock < medium_threshold:
            st.info(f"ℹ️ Average stock is medium: {avg_stock:.0f} units")
        else:
            st.success(f"✅ Average stock is high: {avg_stock:.0f} units")

        # Optional: per-category alerts
        if 'category' in df.columns:
            stock_by_category = df.groupby('category')['on_hand'].mean().reset_index()
            for _, row in stock_by_category.iterrows():
                category = row['category'].title()
                stock = row['on_hand']
                if stock < low_threshold:
                    st.warning(f"⚠️ {category} stock is low: {stock:.0f} units")
                elif stock < medium_threshold:
                    st.info(f"ℹ️ {category} stock is medium: {stock:.0f} units")
                else:
                    st.success(f"✅ {category} stock is high: {stock:.0f} units")

    

# ======================================================
# 4️⃣ Visualizations Page (All Plots)
# ======================================================
if page_option == "Visualizations":
    st.title("📊 Warehouse Data Visualizations")

    # Tabs for plots
    tabs = st.tabs([
        "Category Pie", "Demand Histogram", "Unit Cost Box", "Average Demand Bar",
        "Item Count", "Correlation Heatmap", "Daily Demand Line", "Demand vs Stock Scatter",
        "Demand Violin", "Pairplot"
    ])

    # 1️⃣ PIE CHART
    with tabs[0]:
        if 'category' in df.columns:
            fig = px.pie(df, names='category', title='Category Distribution')
            st.plotly_chart(fig, use_container_width=True)

    # 2️⃣ HISTOGRAM
    with tabs[1]:
        if 'demand' in df.columns:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.histplot(df['demand'], bins=30, kde=True, color='skyblue', ax=ax)
            ax.set_title('Demand Distribution')
            st.pyplot(fig)

    # 3️⃣ BOX PLOT
    with tabs[2]:
        if {'category','unit_cost'}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(9,5))
            sns.boxplot(x='category', y='unit_cost', data=df, palette='Set3', ax=ax)
            ax.set_title('Unit Cost by Category')
            st.pyplot(fig)

    # 4️⃣ BAR CHART
    with tabs[3]:
        if {'category','demand'}.issubset(df.columns):
            avg_demand = df.groupby('category')['demand'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(9,5))
            sns.barplot(x=avg_demand.index, y=avg_demand.values, palette='coolwarm', ax=ax)
            ax.set_title('Average Demand per Category')
            st.pyplot(fig)

    # 5️⃣ COUNT PLOT
    with tabs[4]:
        if 'category' in df.columns:
            fig, ax = plt.subplots(figsize=(9,5))
            sns.countplot(x='category', data=df, order=df['category'].value_counts().index, palette='muted', ax=ax)
            ax.set_title('Item Count by Category')
            st.pyplot(fig)

    # 6️⃣ HEATMAP
    with tabs[5]:
        numeric_cols = df.select_dtypes(include=['float64','int64'])
        if len(numeric_cols.columns) >= 2:
            fig, ax = plt.subplots(figsize=(7,6))
            sns.heatmap(numeric_cols.corr(), annot=True, cmap='YlGnBu', fmt=".2f", ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)

    # 7️⃣ LINE CHART
    with tabs[6]:
        if {'date','demand'}.issubset(df.columns):
            daily_demand = df.groupby('date')['demand'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(12,6))
            sns.lineplot(x='date', y='demand', data=daily_demand, color='green', ax=ax)
            ax.set_title('Daily Demand Trend Over Time')
            st.pyplot(fig)

    # 8️⃣ SCATTER PLOT
    with tabs[7]:
        if {'on_hand','demand'}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(7,5))
            sns.scatterplot(x='on_hand', y='demand', data=df, color='purple', alpha=0.6, ax=ax)
            ax.set_title('Demand vs On-Hand Stock')
            st.pyplot(fig)

    # 9️⃣ VIOLIN PLOT
    with tabs[8]:
        if {'category','demand'}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(9,5))
            sns.violinplot(x='category', y='demand', data=df, palette='husl', ax=ax)
            ax.set_title('Demand Distribution per Category')
            st.pyplot(fig)

    # 🔟 PAIRPLOT
    with tabs[9]:
        selected_cols = ['demand','on_hand','lead_time_days','unit_cost']
        selected_cols = [col for col in selected_cols if col in df.columns]
        if len(selected_cols) >= 2:
            fig = sns.pairplot(df[selected_cols], diag_kind='kde')
            st.pyplot(fig)

# ======================================================
# 5️⃣ ML Model Page (Updated)
# ======================================================
if page_option == "ML Model":
    st.title("🚀 Random Forest Demand Prediction")

    # Feature Engineering
    df['stock_to_cost_ratio'] = df['on_hand'] / (df['unit_cost'] + 1)
    df['lead_time_efficiency'] = df['on_hand'] / (df['lead_time_days'] + 1)
    df['cost_efficiency'] = df['demand'] / (df['unit_cost'] + 1)

    # Features and target
    features = ['on_hand','lead_time_days','unit_cost','stock_to_cost_ratio','lead_time_efficiency','cost_efficiency']
    target = 'demand'

    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Manual MAPE calculation

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")
    col4.metric("R² Score", f"{r2:.3f}")
    col5.metric("MAPE", f"{mape:.2f}%")

    # Feature Importance
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6,4))
    importance.plot(kind='barh', ax=ax, color='teal')
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # Residual Plot
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color='red', ax=ax)
    ax.axhline(0, linestyle='--', color='black')
    ax.set_xlabel("Predicted Demand")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.set_title("Residual Plot")
    st.pyplot(fig)

    # Export predictions
    export_df = X_test.copy()
    export_df['Actual_Demand'] = y_test
    export_df['Predicted_Demand'] = y_pred
    st.download_button("Download Predictions CSV", export_df.to_csv(index=False), "predictions.csv")

# ======================================================
# 6️⃣ Time Series Forecast Page
# ======================================================
if page_option == "Time Series Forecast":
    st.title("🔮 Time Series Demand Forecast (Prophet)")

    if {'date','demand'}.issubset(df.columns):
        ts_df = df.groupby('date')['demand'].sum().reset_index()
        ts_df.columns = ['ds','y']

        m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m.fit(ts_df)

        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)

        # Daily forecast plot
        fig1 = m.plot(forecast)
        st.pyplot(fig1)

        # Weekly forecast
        forecast_weekly = forecast[['ds','yhat']].copy()
        forecast_weekly['week'] = forecast_weekly['ds'].dt.to_period('W').apply(lambda r: r.start_time)
        weekly_pred = forecast_weekly.groupby('week')['yhat'].sum().reset_index()
        fig2 = px.line(weekly_pred, x='week', y='yhat', title='Weekly Demand Forecast')
        st.plotly_chart(fig2, use_container_width=True)

        st.write("Next Week Predicted Demand:", int(weekly_pred['yhat'].iloc[-1]))
