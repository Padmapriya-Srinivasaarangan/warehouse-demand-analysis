import streamlit as st
import os



# =======================================================
#   📦 WAREHOUSE DEMAND DASHBOARD (MAIN PAGE)
# =======================================================

# 🔧 Streamlit Page Configuration
st.set_page_config(
    page_title="Warehouse Demand Dashboard",
    page_icon="📦",
    layout="wide",
)

# =======================================================
#   📌 SIDEBAR NAVIGATION
# =======================================================
st.sidebar.title("📦 Warehouse Dashboard")
st.sidebar.markdown("### Navigate through sections")

pages = {
    "📊 Data Visualization": "pages/_1_data_visualization.py",
    "🤖 Random Forest Model": "pages/_2_model.py",
    "🔮 Time Series Forecasting": "pages/_3_time_series_forecasting.py",
    "⚡ Prediction Tool": "pages/_4_prediction_tool.py",
    "📁 Export Reports": "pages/_5_export.py",
    "ℹ️ About Project": None
    
}


page_selection = st.sidebar.radio("Choose Page", list(pages.keys()))

# =======================================================
#   📘 ABOUT PAGE
# =======================================================
if page_selection == "ℹ️ About Project":
    st.title("ℹ️ About the Warehouse Demand Dashboard")

    st.markdown(
        """
        ### 📦 Project Overview  
        This professional dashboard is designed to analyze and forecast **warehouse demand** using  
        advanced statistical and machine learning techniques.

        ### 🔍 Features Included
        - **📊 Data Visualization**  
          Analyze category-wise data, correlations, outliers, and demand distribution.
        - **🤖 Machine Learning Model (Random Forest)**  
          Predict daily demand using trained ML models.
        - **🔮 Time Series Forecasting (Prophet)**  
          Daily & weekly forecasting based on historical trends.
        - **⚡ Prediction Tool**  
          Gives on-the-fly demand predictions using trained models.

        ### 👩‍💻 Developed By  
        **S. Padmapriya**  

        ### 📁 Dataset  
        `cleaned_warehouse_data.csv`
        """
    )

    st.info(
        """
        💡 Navigate using the sidebar to explore different modules of the dashboard.
        """
    )

# =======================================================
#   📄 LOAD OTHER PAGES SAFELY
# =======================================================
else:
    page_path = pages[page_selection]

    if page_path and os.path.exists(page_path):
        try:
            with open(page_path, "r", encoding="utf-8") as f:
                code = f.read()
                exec(code)
        except Exception as e:
            st.error(f"❌ Error loading page: {e}")

    else:
        st.warning("⚠️ Page not found. Please verify the file path.")
