"""
Home.py
========
Streamlit Home Page for Customer Churn Analytics Dashboard.

Author: Full Stack Data Scientist
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 Customer Churn Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Churn Prediction & Retention Strategy</div>', unsafe_allow_html=True)

st.markdown("---")

# Introduction
st.markdown("""
## 🎯 Welcome to the Customer Churn Analytics Dashboard

This application provides **end-to-end churn prediction and analytics** for proactive customer retention.
Built with state-of-the-art machine learning and deployed for real-time inference.
""")

# Features
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h3>🔍 Single Prediction</h3>
        <p>Enter customer details and get instant churn probability with risk categorization.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h3>📦 Batch Prediction</h3>
        <p>Upload CSV file with multiple customers and download predictions for all.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <h3>📈 Analytics Dashboard</h3>
        <p>Visualize churn risk distribution, top high-risk customers, and retention actions.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# How It Works
st.markdown("## 🚀 How It Works")

st.markdown("""
1. **Data Collection**: Customer behavior, subscription, and interaction data
2. **Feature Engineering**: Create predictive features from raw data
3. **ML Model**: Trained on historical data with multiple algorithms
4. **Prediction API**: FastAPI endpoint for real-time inference
5. **Dashboard**: Streamlit interface for business users
6. **Retention Strategy**: Rule-based recommendations for at-risk customers
""")

# Model Information
with st.expander("🤖 Model Information"):
    st.markdown("""
    **Model Type**: Ensemble (Random Forest / Gradient Boosting)
    
    **Training Data**: 5,000 customers with 17 features
    
    **Performance Metrics**:
    - ROC-AUC: ~0.85
    - Precision: ~0.78
    - Recall: ~0.72
    - F1-Score: ~0.75
    
    **Key Features**:
    - Session activity (last 30 days)
    - Payment history
    - Support interactions
    - Subscription details
    - Demographics
    """)

# Configuration
with st.expander("⚙️ Configuration"):
    st.markdown("""
    **API Configuration**:
    
    Default API URL: `http://localhost:8000`
    
    You can change this in the sidebar settings if your API is running on a different host/port.
    
    **Risk Categories**:
    - Very High: Churn probability ≥ 70%
    - High: Churn probability ≥ 50%
    - Medium: Churn probability ≥ 30%
    - Low: Churn probability < 30%
    """)

# Navigation
st.markdown("---")
st.markdown("## 📱 Navigation")
st.info("""
👈 **Use the sidebar** to navigate between pages:
- **Single Prediction**: Predict churn for one customer
- **Batch Prediction**: Upload CSV for bulk predictions
- **Analytics Dashboard**: View insights and visualizations
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    Built with ❤️ using Streamlit, FastAPI, and scikit-learn<br>
    © 2026 Customer Churn Analytics | Data Science Portfolio Project
</div>
""", unsafe_allow_html=True)
