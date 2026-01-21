"""
1_Single_Prediction.py
======================
Streamlit page for single customer churn prediction.

Author: Full Stack Data Scientist
"""

import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="Single Prediction",
    page_icon="🔍",
    layout="wide"
)

# API Configuration
API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")

st.title("🔍 Single Customer Prediction")
st.markdown("Enter customer details to predict churn probability.")

st.markdown("---")

# Input Form
with st.form("prediction_form"):
    st.subheader("📋 Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        location = st.text_input("Location", value="Mumbai")
        device_type = st.selectbox("Device Type", ["Android", "iOS", "Web"])
        acquisition_channel = st.selectbox("Acquisition Channel", 
                                          ["Organic", "Ads", "Referral", "Partner"])
    
    with col2:
        plan_type = st.selectbox("Plan Type", ["Basic", "Standard", "Premium"])
        monthly_price = st.number_input("Monthly Price", min_value=0.0, value=499.0, step=50.0)
        auto_renew = st.selectbox("Auto Renewal", [0, 1])
        total_sessions_30d = st.number_input("Sessions (30d)", min_value=0.0, value=45.0)
        avg_session_minutes_30d = st.number_input("Avg Session Minutes (30d)", 
                                                   min_value=0.0, value=25.5)
    
    with col3:
        total_crashes_30d = st.number_input("Crashes (30d)", min_value=0.0, value=2.0)
        failed_payments_30d = st.number_input("Failed Payments (30d)", 
                                               min_value=0.0, value=0.0)
        total_amount_success_30d = st.number_input("Successful Payment Amount (30d)", 
                                                    min_value=0.0, value=499.0)
        support_tickets_30d = st.number_input("Support Tickets (30d)", 
                                               min_value=0.0, value=1.0)
        avg_resolution_time_30d = st.number_input("Avg Resolution Time (30d hrs)", 
                                                   min_value=0.0, value=12.5)
    
    submitted = st.form_submit_button("🎯 Predict Churn", use_container_width=True)

# Make Prediction
if submitted:
    # Prepare request data
    customer_data = {
        "age": int(age),
        "gender": gender,
        "location": location,
        "device_type": device_type,
        "acquisition_channel": acquisition_channel,
        "plan_type": plan_type,
        "monthly_price": float(monthly_price),
        "auto_renew": int(auto_renew),
        "total_sessions_30d": float(total_sessions_30d),
        "avg_session_minutes_30d": float(avg_session_minutes_30d),
        "total_crashes_30d": float(total_crashes_30d),
        "failed_payments_30d": float(failed_payments_30d),
        "total_amount_success_30d": float(total_amount_success_30d),
        "support_tickets_30d": float(support_tickets_30d),
        "avg_resolution_time_30d": float(avg_resolution_time_30d)
    }
    
    # Call API
    try:
        with st.spinner("Predicting..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=customer_data,
                headers={"Content-Type": "application/json"}
            )
        
        if response.status_code == 200:
            result = response.json()
            
            st.markdown("---")
            st.success("✅ Prediction Complete!")
            
            # Display Results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Churn Probability", f"{result['churn_probability']*100:.1f}%")
            
            with col2:
                prediction_text = "Will Churn ❌" if result['churn_prediction'] == 1 else "Will Retain ✅"
                st.metric("Prediction", prediction_text)
            
            with col3:
                risk_color = {
                    "Very High": "🔴",
                    "High": "🟠",
                    "Medium": "🟡",
                    "Low": "🟢"
                }
                risk_icon = risk_color.get(result['risk_category'], "⚪")
                st.metric("Risk Category", f"{risk_icon} {result['risk_category']}")
            
            # Recommendation
            st.markdown("---")
            st.subheader("💡 Recommended Actions")
            
            if result['risk_category'] == "Very High":
                st.error("""
                **Immediate Action Required!**
                - Schedule retention call within 24 hours
                - Offer personalized discount (15-25%)
                - Investigate recent issues or complaints
                - Assign priority support representative
                """)
            elif result['risk_category'] == "High":
                st.warning("""
                **High Priority Retention**
                - Send personalized retention offer
                - Survey for feedback
                - Offer plan upgrade/downgrade options
                - Monitor closely for next 7 days
                """)
            elif result['risk_category'] == "Medium":
                st.info("""
                **Engagement Campaign**
                - Send re-engagement content
                - Highlight new features
                - Offer limited-time promotion
                - Monitor activity
                """)
            else:
                st.success("""
                **Continue Standard Engagement**
                - Regular communication
                - Loyalty program updates
                - Monitor for changes
                """)
            
            # Show raw response
            with st.expander("📄 View Raw API Response"):
                st.json(result)
        
        else:
            st.error(f"❌ API Error: {response.status_code}")
            st.error(response.text)
    
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_URL}")
        st.info("Make sure the FastAPI server is running: `uvicorn app.main:app --reload`")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Example
with st.expander("💡 Example Customer Profiles"):
    st.markdown("""
    **Low Risk Profile:**
    - Age: 28, Premium plan, Auto-renewal ON
    - 85 sessions, 45 min avg, 0 failed payments
    - 0 support tickets
    
    **High Risk Profile:**
    - Age: 45, Basic plan, Auto-renewal OFF
    - 5 sessions, 8 min avg, 2 failed payments
    - 4 support tickets, long resolution times
    """)
