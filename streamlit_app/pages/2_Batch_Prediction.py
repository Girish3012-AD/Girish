"""
2_Batch_Prediction.py
=====================
Streamlit page for batch churn prediction via CSV upload.

Author: Full Stack Data Scientist
"""

import streamlit as st
import pandas as pd
import requests
import io

# Page config
st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📦",
    layout="wide"
)

# API Configuration
API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")

st.title("📦 Batch Customer Prediction")
st.markdown("Upload a CSV file with customer data to get predictions for all customers.")

st.markdown("---")

# CSV Template
st.subheader("📥 CSV Template")
st.info("""
Your CSV must contain these columns:
`age, gender, location, device_type, acquisition_channel, plan_type, monthly_price, auto_renew, 
total_sessions_30d, avg_session_minutes_30d, total_crashes_30d, failed_payments_30d, 
total_amount_success_30d, support_tickets_30d, avg_resolution_time_30d`
""")

# Download template
template_data = {
    'age': [35, 45, 28],
    'gender': ['Male', 'Female', 'Other'],
    'location': ['Mumbai', 'Delhi', 'Bangalore'],
    'device_type': ['Android', 'iOS', 'Web'],
    'acquisition_channel': ['Organic', 'Ads', 'Referral'],
    'plan_type': ['Standard', 'Basic', 'Premium'],
    'monthly_price': [499.0, 199.0, 999.0],
    'auto_renew': [1, 0, 1],
    'total_sessions_30d': [45, 5, 85],
    'avg_session_minutes_30d': [25.5, 8.5, 45.2],
    'total_crashes_30d': [2, 3, 1],
    'failed_payments_30d': [0, 2, 0],
    'total_amount_success_30d': [499.0, 199.0, 999.0],
    'support_tickets_30d': [1, 4, 0],
    'avg_resolution_time_30d': [12.5, 48.5, 0.0]
}
template_df = pd.DataFrame(template_data)

csv_template = template_df.to_csv(index=False)
st.download_button(
    label="📥 Download CSV Template",
    data=csv_template,
    file_name="customer_template.csv",
    mime="text/csv"
)

st.markdown("---")

# File Upload
st.subheader("📤 Upload Your File")
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! {len(df)} customers found.")
        
        # Show preview
        with st.expander("👀 Preview Data"):
            st.dataframe(df.head(10))
        
        # Predict button
        if st.button("🎯 Predict for All Customers", use_container_width=True):
            with st.spinner(f"Predicting for {len(df)} customers..."):
                # Convert to list of dicts
                customers = df.to_dict('records')
                
                # Prepare request
                batch_request = {"customers": customers}
                
                try:
                    response = requests.post(
                        f"{API_URL}/batch_predict",
                        json=batch_request,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Predictions Complete!")
                        
                        # Display Summary
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Customers", result['total_customers'])
                        
                        with col2:
                            st.metric("High Risk", result['high_risk_count'])
                        
                        with col3:
                            high_risk_pct = (result['high_risk_count'] / result['total_customers']) * 100
                            st.metric("High Risk %", f"{high_risk_pct:.1f}%")
                        
                        st.markdown("---")
                        
                        # Create results dataframe
                        predictions = result['predictions']
                        results_df = df.copy()
                        results_df['churn_probability'] = [p['churn_probability'] for p in predictions]
                        results_df['churn_prediction'] = [p['churn_prediction'] for p in predictions]
                        results_df['risk_category'] = [p['risk_category'] for p in predictions]
                        
                        # Show results
                        st.subheader("📊 Prediction Results")
                        
                        # Filter options
                        col1, col2 = st.columns(2)
                        with col1:
                            risk_filter = st.multiselect(
                                "Filter by Risk Category",
                                ["Low", "Medium", "High", "Very High"],
                                default=["High", "Very High"]
                            )
                        
                        if risk_filter:
                            filtered_df = results_df[results_df['risk_category'].isin(risk_filter)]
                        else:
                            filtered_df = results_df
                        
                        # Display filtered results
                        st.dataframe(
                            filtered_df.sort_values('churn_probability', ascending=False),
                            use_container_width=True
                        )
                        
                        # Download results
                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv_results,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Visualization
                        st.markdown("---")
                        st.subheader("📈 Risk Distribution")
                        
                        risk_counts = results_df['risk_category'].value_counts()
                        st.bar_chart(risk_counts)
                    
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        st.error(response.text)
                
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {API_URL}")
                    st.info("Make sure the FastAPI server is running")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Error reading CSV: {str(e)}")

# Notes
with st.expander("ℹ️ Important Notes"):
    st.markdown("""
    - Maximum 1000 customers per batch
    - All columns must be present in the CSV
    - Missing values will be filled with defaults
    - Results are not saved automatically - download the CSV
    """)
