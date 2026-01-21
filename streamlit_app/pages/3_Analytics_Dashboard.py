"""
3_Analytics_Dashboard.py
========================
Streamlit analytics dashboard for churn insights.

Author: Full Stack Data Scientist
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Try plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Churn Analytics Dashboard")
st.markdown("Visualize customer risk scores and retention actions.")

st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load customer scores and retention actions."""
    project_root = Path(__file__).parent.parent.parent
    outputs_dir = project_root / 'outputs'
    
    try:
        scores_df = pd.read_csv(outputs_dir / 'customer_scores.csv')
        actions_df = pd.read_csv(outputs_dir / 'retention_actions.csv')
        return scores_df, actions_df, None
    except FileNotFoundError as e:
        return None, None, str(e)

scores_df, actions_df, error = load_data()

if error:
    st.error(f"❌ Cannot load data: {error}")
    st.info("""
    To generate the required data files, run:
    1. Day 4 notebook: `04_explainability_segmentation.ipynb`
    2. This will create `customer_scores.csv` and `retention_actions.csv`
    """)
    st.stop()

st.success(f"✅ Loaded {len(scores_df):,} customer scores")

# KPI Cards
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_prob = scores_df['churn_probability'].mean()
    st.metric("Avg Churn Probability", f"{avg_prob*100:.1f}%")

with col2:
    high_risk = (scores_df['churn_probability'] >= 0.7).sum()
    st.metric("High Risk Customers", f"{high_risk:,}")

with col3:
    predicted_churners = scores_df['churn_prediction'].sum()
    st.metric("Predicted Churners", f"{predicted_churners:,}")

with col4:
    churn_rate = (predicted_churners / len(scores_df)) * 100
    st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")

st.markdown("---")

# Churn Probability Distribution
st.subheader("📊 Churn Probability Distribution")

if PLOTLY_AVAILABLE:
    fig = px.histogram(
        scores_df, 
        x='churn_probability',
        nbins=50,
        title='Distribution of Churn Probabilities',
        labels={'churn_probability': 'Churn Probability'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores_df['churn_probability'], bins=50, color='#1f77b4', edgecolor='black')
    ax.set_xlabel('Churn Probability')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Churn Probabilities')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.markdown("---")

# Top High-Risk Customers
st.subheader("🚨 Top 20 High-Risk Customers")

top_risk = scores_df.nlargest(20, 'churn_probability')[
    ['customer_id', 'churn_probability', 'churn_prediction', 'risk_category']
]

# Add retention actions if available
if actions_df is not None and 'recommended_action' in actions_df.columns:
    top_risk = top_risk.merge(
        actions_df[['customer_id', 'recommended_action']], 
        on='customer_id', 
        how='left'
    )

# Format probability as percentage
top_risk_display = top_risk.copy()
top_risk_display['churn_probability'] = top_risk_display['churn_probability'].apply(
    lambda x: f"{x*100:.1f}%"
)

st.dataframe(top_risk_display, use_container_width=True)

st.markdown("---")

# Risk Category Distribution
st.subheader("🎯 Risk Category Breakdown")

col1, col2 = st.columns([2, 1])

with col1:
    if PLOTLY_AVAILABLE:
        risk_counts = scores_df['risk_category'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Risk Category Distribution',
            color_discrete_sequence=px.colors.sequential.RdBu_r
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        risk_counts = scores_df['risk_category'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title('Risk Category Distribution')
        st.pyplot(fig)

with col2:
    st.write("**Risk Counts:**")
    for category in ['Low', 'Medium', 'High', 'Very High']:
        count = (scores_df['risk_category'] == category).sum()
        pct = (count / len(scores_df)) * 100
        st.write(f"**{category}**: {count:,} ({pct:.1f}%)")

st.markdown("---")

# Retention Actions
if actions_df is not None and 'recommended_action' in actions_df.columns:
    st.subheader("💡 Recommended Actions Distribution")
    
    action_counts = actions_df['recommended_action'].value_counts().head(10)
    
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            x=action_counts.index,
            y=action_counts.values,
            title='Top 10 Retention Actions',
            labels={'x': 'Action', 'y': 'Count'},
            color=action_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(action_counts)), action_counts.values, color='#3498db')
        ax.set_yticks(range(len(action_counts)))
        ax.set_yticklabels(action_counts.index)
        ax.invert_yaxis()
        ax.set_xlabel('Count')
        ax.set_title('Top 10 Retention Actions')
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

st.markdown("---")

# Segment Analysis (if available)
if 'segment_id' in actions_df.columns:
    st.subheader("🎯 Segment-wise Churn Risk")
    
    segment_risk = actions_df.groupby('segment_id')['churn_probability'].agg(['mean', 'count'])
    
    segment_risk.columns = ['Avg Churn Probability', 'Customer Count']
    segment_risk.index.name = 'Segment'
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Segment Statistics:**")
        st.dataframe(segment_risk)
    
    with col2:
        if PLOTLY_AVAILABLE:
            fig = px.bar(
                x=segment_risk.index,
                y=segment_risk['Avg Churn Probability'],
                title='Avg Churn Probability by Segment',
                labels={'x': 'Segment', 'y': 'Avg Churn Probability'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Export Options
st.markdown("---")
st.subheader("📥 Export Data")

col1, col2 = st.columns(2)

with col1:
    csv_scores = scores_df.to_csv(index=False)
    st.download_button(
        label="Download Customer Scores CSV",
        data=csv_scores,
        file_name="customer_scores_export.csv",
        mime="text/csv"
    )

with col2:
    if actions_df is not None:
        csv_actions = actions_df.to_csv(index=False)
        st.download_button(
            label="Download Retention Actions CSV",
            data=csv_actions,
            file_name="retention_actions_export.csv",
            mime="text/csv"
        )
