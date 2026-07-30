# ======================================================
# 📊 WAREHOUSE INVENTORY DATA VISUALIZATION DASHBOARD
# ======================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st




st.title("📊 Warehouse Inventory Data Visualization Dashboard")
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




# Load Data
df = pd.read_csv("cleaned_warehouse_data.csv")

# -----------------------------
# 🥧 Pie Chart - Category Share
# -----------------------------
st.subheader("🥧 Category Distribution")
category_counts = df['category'].value_counts()
fig, ax = plt.subplots(figsize=(6,6))
ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
st.pyplot(fig)

st.markdown("""
**📌 What this chart shows:**
- Displays the share of each product category in the dataset.  
- Helps understand which category is dominant.  
- Useful for inventory distribution analysis.
""")

# -----------------------------
# 📊 Histogram - Demand Distribution
# -----------------------------
st.subheader("📊 Demand Distribution")
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(df['demand'], bins=30, color='skyblue', kde=True, ax=ax)
st.pyplot(fig)

st.markdown("""
**📌 What this chart shows:**
- Distribution of demand values across all items.  
- Identifies whether demand is low, medium, or highly skewed.  
- The KDE curve helps understand overall demand trends.
""")

# -----------------------------
# 📦 Box Plot - Unit Cost per Category
# -----------------------------
st.subheader("📦 Unit Cost by Category")
fig, ax = plt.subplots(figsize=(8,4))
sns.boxplot(x='category', y='unit_cost', data=df, palette='Set3', ax=ax)
st.pyplot(fig)

st.markdown("""
**📌 What this chart shows:**
- Shows cost variation inside each category.  
- Helps detect high-cost items and outliers.  
- Useful for identifying categories with large price differences.
""")

# -----------------------------
# 📈 Bar Chart - Average Demand per Category
# -----------------------------
st.subheader("📈 Average Demand by Category")
avg_demand = df.groupby('category')['demand'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=avg_demand.index, y=avg_demand.values, palette='coolwarm', ax=ax)
st.pyplot(fig)

st.markdown("""
**📌 What this chart shows:**
- Displays average demand for each category.  
- Helps find high-performing categories.  
- Useful for demand-based stocking decisions.
""")

# -----------------------------
# 🔢 Count Plot - Item Count per Category
# -----------------------------
st.subheader("🔢 Item Count by Category")
fig, ax = plt.subplots(figsize=(8,4))
sns.countplot(x='category', data=df, order=df['category'].value_counts().index, palette='muted', ax=ax)
st.pyplot(fig)

st.markdown("""
**📌 What this chart shows:**
- Shows how many unique items each category contains.  
- Helps understand category quantity distribution.  
- Useful for inventory size comparison.
""")

# -----------------------------
# 🔥 Heatmap - Correlation
# -----------------------------
st.subheader("🔥 Correlation Heatmap")
numeric_cols = df.select_dtypes(include=['int64','float64'])
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='YlGnBu', fmt=".2f", ax=ax)
st.pyplot(fig)

st.markdown("""
**📌 What this chart shows:**
- Displays relationships between numeric features.  
- Helps identify strong positive or negative correlations.  
- Useful for feature selection in machine learning.
""")

# -----------------------------
# 📆 Line Chart - Demand Over Time
# -----------------------------
st.subheader("📆 Demand Over Time")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
daily_demand = df.groupby('date')['demand'].sum().reset_index()
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(daily_demand['date'], daily_demand['demand'], color='blue', marker='o')
ax.set_xlabel("Date")
ax.set_ylabel("Demand")
ax.set_title("Daily Demand Over Time")
ax.grid(True)
st.pyplot(fig)

st.markdown("""
**📌 What this chart shows:**
- Shows how total demand changes over time.  
- Helps detect seasonality, trends, and anomalies.  
- Useful for forecasting and planning stock.
""")

# -----------------------------
# ⚪ Scatter Plot - Demand vs On-Hand Stock
# -----------------------------
st.subheader("⚪ Demand vs On-Hand Stock")
fig, ax = plt.subplots(figsize=(8,4))
sns.scatterplot(x='on_hand', y='demand', data=df, hue='category', palette='Set2', ax=ax)
ax.set_xlabel("On-Hand Stock")
ax.set_ylabel("Demand")
st.pyplot(fig)

st.markdown("""
**📌 What this chart shows:**
- Relationship between available stock and demand.  
- Helps identify whether more stock leads to more demand.  
- Categories are color-coded for easy comparison.
""")

# -----------------------------
# 🎻 Violin Plot - Category-wise Demand Density
# -----------------------------
st.subheader("🎻 Category-wise Demand Density")
fig, ax = plt.subplots(figsize=(8,4))
sns.violinplot(x='category', y='demand', data=df, palette='Pastel1', ax=ax)
st.pyplot(fig)

st.markdown("""
**📌 What this chart shows:**
- Shows how demand values are spread within each category.  
- Highlights density, spread, and skewness of demand.  
- Useful for understanding demand behavior.
""")

# -----------------------------
# 🧩 Pairplot - Overall Variable Relationships
# -----------------------------
st.subheader("🧩 Pairplot of Numeric Features")
numeric_features = df.select_dtypes(include=['int64','float64'])
fig = sns.pairplot(numeric_features)
st.pyplot(fig)

st.markdown("""
**📌 What this chart shows:**
- Displays pairwise relationships between all numeric variables.  
- Helps identify patterns, trends, and linear relations.  
- Useful for exploratory data analysis (EDA).
""")
