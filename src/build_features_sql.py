"""
build_features_sql.py
=====================
SQL Queries for Feature Engineering.

Contains all SQL queries used to build the ML-ready dataset.
Aggregations are computed for the last 30 days from global max dates.

Author: Senior Data Engineer + Data Scientist
"""


# =============================================================================
# QUERIES TO GET MAX DATES
# =============================================================================

QUERY_MAX_EVENT_DATE = """
SELECT MAX(event_date) as max_date FROM app_events_daily;
"""

QUERY_MAX_TXN_DATE = """
SELECT MAX(txn_date) as max_date FROM transactions;
"""

QUERY_MAX_TICKET_DATE = """
SELECT MAX(ticket_date) as max_date FROM support_tickets;
"""


# =============================================================================
# MAIN ML DATASET QUERY
# =============================================================================

def get_ml_dataset_query(max_event_date: str, max_txn_date: str, max_ticket_date: str) -> str:
    """
    Build the main SQL query for ML dataset.
    
    Aggregations are computed for last 30 days from the global max date of each table.
    
    Args:
        max_event_date: Max date from app_events_daily (YYYY-MM-DD)
        max_txn_date: Max date from transactions (YYYY-MM-DD)
        max_ticket_date: Max date from support_tickets (YYYY-MM-DD)
    
    Returns:
        SQL query string
    """
    
    return f"""
    WITH 
    -- =======================================================================
    -- App events aggregation (last 30 days from max event_date)
    -- =======================================================================
    app_agg AS (
        SELECT 
            customer_id,
            COALESCE(SUM(sessions), 0) as total_sessions_30d,
            COALESCE(AVG(avg_session_minutes), 0) as avg_session_minutes_30d,
            COALESCE(SUM(crashes), 0) as total_crashes_30d
        FROM app_events_daily
        WHERE event_date >= date('{max_event_date}', '-30 days')
          AND event_date <= '{max_event_date}'
        GROUP BY customer_id
    ),
    
    -- =======================================================================
    -- Transaction aggregation (last 30 days from max txn_date)
    -- - failed_payments_30d: COUNT where payment_status = 'failed'
    -- - total_amount_success_30d: SUM(amount) where payment_status = 'success'
    -- =======================================================================
    txn_agg AS (
        SELECT 
            customer_id,
            COALESCE(SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END), 0) as failed_payments_30d,
            COALESCE(SUM(CASE WHEN payment_status = 'success' THEN amount ELSE 0 END), 0) as total_amount_success_30d
        FROM transactions
        WHERE txn_date >= date('{max_txn_date}', '-30 days')
          AND txn_date <= '{max_txn_date}'
        GROUP BY customer_id
    ),
    
    -- =======================================================================
    -- Support tickets aggregation (last 30 days from max ticket_date)
    -- =======================================================================
    ticket_agg AS (
        SELECT 
            customer_id,
            COALESCE(COUNT(*), 0) as support_tickets_30d,
            COALESCE(AVG(resolution_time_hours), 0) as avg_resolution_time_30d
        FROM support_tickets
        WHERE ticket_date >= date('{max_ticket_date}', '-30 days')
          AND ticket_date <= '{max_ticket_date}'
        GROUP BY customer_id
    )
    
    -- =======================================================================
    -- Main query: Join all tables
    -- =======================================================================
    SELECT 
        -- Customer demographics
        c.customer_id,
        c.age,
        c.gender,
        c.location,
        c.device_type,
        c.acquisition_channel,
        
        -- Subscription info
        s.plan_type,
        s.monthly_price,
        COALESCE(s.auto_renew, 0) as auto_renew,
        
        -- App usage features (30-day window)
        COALESCE(a.total_sessions_30d, 0) as total_sessions_30d,
        ROUND(COALESCE(a.avg_session_minutes_30d, 0), 2) as avg_session_minutes_30d,
        COALESCE(a.total_crashes_30d, 0) as total_crashes_30d,
        
        -- Transaction features (30-day window)
        COALESCE(t.failed_payments_30d, 0) as failed_payments_30d,
        ROUND(COALESCE(t.total_amount_success_30d, 0), 2) as total_amount_success_30d,
        
        -- Support ticket features (30-day window)
        COALESCE(tk.support_tickets_30d, 0) as support_tickets_30d,
        ROUND(COALESCE(tk.avg_resolution_time_30d, 0), 2) as avg_resolution_time_30d,
        
        -- Target variable: churn label from churn_labels table
        cl.churn
        
    FROM customers c
    
    -- Join subscriptions
    LEFT JOIN subscriptions s 
        ON c.customer_id = s.customer_id
    
    -- Join churn labels
    LEFT JOIN churn_labels cl 
        ON c.customer_id = cl.customer_id
    
    -- Join aggregated features
    LEFT JOIN app_agg a 
        ON c.customer_id = a.customer_id
    LEFT JOIN txn_agg t 
        ON c.customer_id = t.customer_id
    LEFT JOIN ticket_agg tk 
        ON c.customer_id = tk.customer_id
    
    -- Ensure no duplicates (one row per customer)
    GROUP BY c.customer_id
    
    ORDER BY c.customer_id;
    """


# =============================================================================
# ALTERNATIVE: Single query to get all max dates at once
# =============================================================================

QUERY_ALL_MAX_DATES = """
SELECT 
    (SELECT MAX(event_date) FROM app_events_daily) as max_event_date,
    (SELECT MAX(txn_date) FROM transactions) as max_txn_date,
    (SELECT MAX(ticket_date) FROM support_tickets) as max_ticket_date;
"""


# =============================================================================
# UTILITY: Print queries for debugging
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SQL FEATURE ENGINEERING QUERIES")
    print("=" * 60)
    
    print("\n>>> Max Dates Query:")
    print(QUERY_ALL_MAX_DATES)
    
    print("\n>>> ML Dataset Query (sample with placeholder dates):")
    sample_query = get_ml_dataset_query('2025-12-31', '2025-12-31', '2025-12-31')
    print(sample_query)
