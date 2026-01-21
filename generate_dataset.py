"""
generate_dataset.py
====================
Synthetic Data Generator for Customer Churn Analytics Project

Generates realistic multi-table dataset with:
- 5000 customers with realistic demographics
- 12 months of daily activity data
- Churn rate ~15-30% with class imbalance
- Noise, missing values, and referential integrity

Author: Senior Data Engineer + Data Scientist
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 42
NUM_CUSTOMERS = 5000
CHURN_RATE_TARGET = 0.22  # ~22% churn rate
MISSING_VALUE_RATE = 0.02  # ~2% missing values

# Date range: 12 months of data
END_DATE = datetime(2025, 12, 31)
START_DATE = END_DATE - timedelta(days=365)

# Indian cities for location
CITIES = [
    'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai',
    'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow',
    'Surat', 'Kanpur', 'Nagpur', 'Indore', 'Thane',
    'Bhopal', 'Visakhapatnam', 'Patna', 'Vadodara', 'Ghaziabad'
]

# Plan configuration
PLANS = {
    'Basic': {'price': 199, 'churn_prob_multiplier': 1.5},
    'Standard': {'price': 499, 'churn_prob_multiplier': 1.0},
    'Premium': {'price': 999, 'churn_prob_multiplier': 0.5}
}

# Content categories
CONTENT_CATEGORIES = ['Sports', 'Movies', 'Series', 'Education', 'News']

# Issue types for support tickets
ISSUE_TYPES = ['Billing', 'Network', 'AppBug', 'Refund', 'Account']

# Churn reasons
CHURN_REASONS = [
    'Price too high', 'Poor content quality', 'Technical issues',
    'Switched to competitor', 'Financial constraints', 'Not using enough',
    'Bad customer service', 'Better offer elsewhere'
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def generate_customer_id(index: int) -> str:
    """Generate customer ID in format C000001."""
    return f"C{index:06d}"


def generate_txn_id(index: int) -> str:
    """Generate transaction ID in format TXN000001."""
    return f"TXN{index:08d}"


def generate_ticket_id(index: int) -> str:
    """Generate ticket ID in format TKT000001."""
    return f"TKT{index:06d}"


def random_date_between(start: datetime, end: datetime) -> datetime:
    """Generate random date between start and end."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)


def add_missing_values(df: pd.DataFrame, columns: list, rate: float = MISSING_VALUE_RATE) -> pd.DataFrame:
    """Add random missing values to specified columns."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            mask = np.random.random(len(df)) < rate
            df.loc[mask, col] = np.nan
    return df


# =============================================================================
# DATA GENERATORS
# =============================================================================

def generate_customers(num_customers: int = NUM_CUSTOMERS) -> pd.DataFrame:
    """
    Generate customer demographic data.
    
    Returns DataFrame with: customer_id, age, gender, location, signup_date,
                           device_type, acquisition_channel
    """
    print("Generating customers...")
    
    customer_ids = [generate_customer_id(i + 1) for i in range(num_customers)]
    
    # Age distribution: weighted towards 25-40 age group
    ages = np.random.choice(
        range(18, 61),
        size=num_customers,
        p=None  # uniform for simplicity
    )
    ages = np.clip(np.random.normal(32, 10, num_customers).astype(int), 18, 60)
    
    # Gender distribution: ~48% Male, ~48% Female, ~4% Other
    genders = np.random.choice(
        ['Male', 'Female', 'Other'],
        size=num_customers,
        p=[0.48, 0.48, 0.04]
    )
    
    # Location: Random Indian cities
    locations = np.random.choice(CITIES, size=num_customers)
    
    # Signup dates: Distributed over last 18 months for variety
    signup_start = START_DATE - timedelta(days=180)
    signup_dates = [
        random_date_between(signup_start, END_DATE - timedelta(days=30))
        for _ in range(num_customers)
    ]
    
    # Device types
    device_types = np.random.choice(
        ['Android', 'iOS', 'Web'],
        size=num_customers,
        p=[0.55, 0.30, 0.15]
    )
    
    # Acquisition channels
    acquisition_channels = np.random.choice(
        ['Organic', 'Ads', 'Referral', 'Partner'],
        size=num_customers,
        p=[0.35, 0.30, 0.20, 0.15]
    )
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'location': locations,
        'signup_date': signup_dates,
        'device_type': device_types,
        'acquisition_channel': acquisition_channels
    })
    
    df['signup_date'] = pd.to_datetime(df['signup_date']).dt.date
    
    return df


def generate_subscriptions(customers_df: pd.DataFrame, churn_decisions: dict) -> pd.DataFrame:
    """
    Generate subscription data linked to customers.
    
    Business rules:
    - Premium users churn less than Basic users
    - Users with failed payments churn more (handled later)
    """
    print("Generating subscriptions...")
    
    records = []
    plan_types = list(PLANS.keys())
    plan_weights = [0.40, 0.35, 0.25]  # Basic, Standard, Premium
    
    for _, customer in customers_df.iterrows():
        customer_id = customer['customer_id']
        signup_date = customer['signup_date']
        
        # Assign plan type (weighted)
        plan_type = np.random.choice(plan_types, p=plan_weights)
        monthly_price = PLANS[plan_type]['price']
        
        # Start date is signup date
        start_date = signup_date
        
        # Determine if churned based on pre-calculated decisions
        is_churned = churn_decisions.get(customer_id, False)
        
        if is_churned:
            # End date: somewhere between signup and end of period
            days_active = random.randint(30, 300)
            end_date = pd.to_datetime(signup_date) + timedelta(days=days_active)
            if end_date > pd.to_datetime(END_DATE):
                end_date = pd.to_datetime(END_DATE) - timedelta(days=random.randint(1, 30))
            end_date = end_date.date()
            is_active = 0
        else:
            end_date = None
            is_active = 1
        
        # Auto-renew: churned users less likely to have auto-renew
        auto_renew = 0 if is_churned else int(np.random.random() > 0.3)
        
        records.append({
            'customer_id': customer_id,
            'plan_type': plan_type,
            'monthly_price': monthly_price,
            'start_date': start_date,
            'end_date': end_date,
            'is_active': is_active,
            'auto_renew': auto_renew
        })
    
    return pd.DataFrame(records)


def generate_transactions(customers_df: pd.DataFrame, subscriptions_df: pd.DataFrame,
                          churn_decisions: dict) -> pd.DataFrame:
    """
    Generate transaction data.
    
    Business rules:
    - Failed payments correlate with churn probability
    - Monthly transactions based on subscription
    """
    print("Generating transactions...")
    
    records = []
    txn_counter = 0
    payment_methods = ['UPI', 'Card', 'NetBanking', 'Wallet']
    payment_weights = [0.40, 0.30, 0.20, 0.10]
    
    for _, sub in subscriptions_df.iterrows():
        customer_id = sub['customer_id']
        monthly_price = sub['monthly_price']
        start_date = pd.to_datetime(sub['start_date'])
        end_date = pd.to_datetime(sub['end_date']) if sub['end_date'] else pd.to_datetime(END_DATE)
        is_churned = churn_decisions.get(customer_id, False)
        
        # Generate monthly transactions
        current_date = start_date
        while current_date <= end_date:
            txn_counter += 1
            
            # Failed payment probability: higher for churned users
            if is_churned:
                fail_prob = 0.15  # 15% fail rate for churners
            else:
                fail_prob = 0.03  # 3% fail rate for non-churners
            
            payment_status = 'failed' if np.random.random() < fail_prob else 'success'
            
            # Add some variation to amount
            amount = monthly_price * (1 + np.random.uniform(-0.05, 0.05))
            
            records.append({
                'txn_id': generate_txn_id(txn_counter),
                'customer_id': customer_id,
                'txn_date': current_date.date(),
                'amount': round(amount, 2),
                'payment_status': payment_status,
                'payment_method': np.random.choice(payment_methods, p=payment_weights)
            })
            
            # Move to next month
            current_date += timedelta(days=30)
    
    return pd.DataFrame(records)


def generate_app_events(customers_df: pd.DataFrame, subscriptions_df: pd.DataFrame,
                        churn_decisions: dict) -> pd.DataFrame:
    """
    Generate daily app usage events.
    
    Business rules:
    - Churning users show declining sessions in last 30 days
    - Random inactive days and noise
    """
    print("Generating app events (this may take a moment)...")
    
    records = []
    
    # Create lookup for subscription dates
    sub_lookup = subscriptions_df.set_index('customer_id').to_dict('index')
    
    for _, customer in customers_df.iterrows():
        customer_id = customer['customer_id']
        sub_info = sub_lookup.get(customer_id, {})
        
        start_date = pd.to_datetime(sub_info.get('start_date', customer['signup_date']))
        end_date = pd.to_datetime(sub_info.get('end_date')) if sub_info.get('end_date') else pd.to_datetime(END_DATE)
        is_churned = churn_decisions.get(customer_id, False)
        
        # Generate daily events
        current_date = start_date
        days_until_churn = (end_date - start_date).days if is_churned else None
        
        while current_date <= min(end_date, pd.to_datetime(END_DATE)):
            # Skip some days randomly (inactive days)
            if np.random.random() < 0.15:
                current_date += timedelta(days=1)
                continue
            
            # Calculate days remaining (for churn behavior)
            days_remaining = (end_date - current_date).days if is_churned else 999
            
            # Base session count
            base_sessions = np.random.randint(1, 15)
            
            # Declining behavior for churners in last 30 days
            if is_churned and days_remaining < 30:
                decline_factor = days_remaining / 30
                sessions = max(0, int(base_sessions * decline_factor * np.random.uniform(0.5, 1.0)))
            else:
                sessions = base_sessions
            
            # Session minutes
            if sessions > 0:
                avg_session_minutes = round(np.random.uniform(5, 60) * (sessions / 10), 2)
            else:
                avg_session_minutes = 0
            
            # Crashes: more for churners
            crash_prob = 0.3 if is_churned else 0.1
            crashes = np.random.poisson(1) if np.random.random() < crash_prob else 0
            crashes = min(crashes, 5)
            
            # Content category
            content_category = np.random.choice(CONTENT_CATEGORIES)
            
            records.append({
                'customer_id': customer_id,
                'event_date': current_date.date(),
                'sessions': sessions,
                'avg_session_minutes': avg_session_minutes,
                'crashes': crashes,
                'content_category': content_category
            })
            
            current_date += timedelta(days=1)
    
    df = pd.DataFrame(records)
    
    # Add some missing values
    df = add_missing_values(df, ['avg_session_minutes'], rate=0.01)
    
    return df


def generate_support_tickets(customers_df: pd.DataFrame, subscriptions_df: pd.DataFrame,
                             churn_decisions: dict) -> pd.DataFrame:
    """
    Generate support ticket data.
    
    Business rules:
    - Churned users have more billing issues
    - Churned users have longer resolution times
    """
    print("Generating support tickets...")
    
    records = []
    ticket_counter = 0
    
    # Create lookup for subscription dates
    sub_lookup = subscriptions_df.set_index('customer_id').to_dict('index')
    
    for _, customer in customers_df.iterrows():
        customer_id = customer['customer_id']
        sub_info = sub_lookup.get(customer_id, {})
        is_churned = churn_decisions.get(customer_id, False)
        
        start_date = pd.to_datetime(sub_info.get('start_date', customer['signup_date']))
        end_date = pd.to_datetime(sub_info.get('end_date')) if sub_info.get('end_date') else pd.to_datetime(END_DATE)
        
        # Number of tickets: churners have more
        if is_churned:
            num_tickets = np.random.poisson(3)  # Average 3 tickets
        else:
            num_tickets = np.random.poisson(1)  # Average 1 ticket
        
        for _ in range(num_tickets):
            ticket_counter += 1
            ticket_date = random_date_between(start_date.to_pydatetime(), end_date.to_pydatetime())
            
            # Issue type: churners more likely to have billing issues
            if is_churned:
                issue_weights = [0.35, 0.20, 0.15, 0.20, 0.10]  # More billing
            else:
                issue_weights = [0.15, 0.25, 0.25, 0.15, 0.20]  # Normal distribution
            
            issue_type = np.random.choice(ISSUE_TYPES, p=issue_weights)
            
            # Resolution time: longer for churners
            if is_churned:
                resolution_time = np.random.uniform(20, 80)
            else:
                resolution_time = np.random.uniform(1, 40)
            
            # Is resolved: lower rate for churners
            is_resolved = 1 if np.random.random() > (0.3 if is_churned else 0.1) else 0
            
            records.append({
                'ticket_id': generate_ticket_id(ticket_counter),
                'customer_id': customer_id,
                'ticket_date': ticket_date.date(),
                'issue_type': issue_type,
                'resolution_time_hours': round(resolution_time, 2),
                'is_resolved': is_resolved
            })
    
    return pd.DataFrame(records)


def generate_churn_labels(customers_df: pd.DataFrame, subscriptions_df: pd.DataFrame,
                          churn_decisions: dict) -> pd.DataFrame:
    """
    Generate churn labels based on subscription status.
    
    Churn definition: churn=1 if is_active=0 OR end_date IS NOT NULL
    """
    print("Generating churn labels...")
    
    records = []
    sub_lookup = subscriptions_df.set_index('customer_id').to_dict('index')
    
    for _, customer in customers_df.iterrows():
        customer_id = customer['customer_id']
        sub_info = sub_lookup.get(customer_id, {})
        
        is_active = sub_info.get('is_active', 1)
        end_date = sub_info.get('end_date')
        
        # Churn = 1 if not active OR has end_date
        churn = 1 if (is_active == 0 or end_date is not None) else 0
        
        churn_date = end_date if churn else None
        churn_reason = np.random.choice(CHURN_REASONS) if churn else None
        
        records.append({
            'customer_id': customer_id,
            'churn': churn,
            'churn_date': churn_date,
            'churn_reason': churn_reason
        })
    
    return pd.DataFrame(records)


def decide_churn(customers_df: pd.DataFrame) -> dict:
    """
    Pre-calculate churn decisions for each customer.
    
    Uses plan type and randomness to determine churn.
    Returns dict mapping customer_id -> bool (is_churned)
    """
    print("Calculating churn decisions...")
    
    churn_decisions = {}
    plan_probs = {'Basic': 0.30, 'Standard': 0.20, 'Premium': 0.12}
    
    for idx, customer in customers_df.iterrows():
        customer_id = customer['customer_id']
        
        # Assign preliminary plan (will be used in subscription generation)
        plan_type = np.random.choice(
            list(plan_probs.keys()),
            p=[0.40, 0.35, 0.25]
        )
        
        # Churn probability based on plan
        base_prob = plan_probs[plan_type]
        
        # Add some noise
        prob = base_prob * np.random.uniform(0.7, 1.3)
        
        # Determine churn
        churn_decisions[customer_id] = np.random.random() < prob
    
    return churn_decisions


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to generate all datasets."""
    
    print("=" * 60)
    print("CUSTOMER CHURN DATASET GENERATOR")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Get project root directory
    project_root = Path(__file__).parent
    raw_data_dir = project_root / 'data' / 'raw'
    
    # Create directories if they don't exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {raw_data_dir}")
    
    # Generate customers first
    customers_df = generate_customers(NUM_CUSTOMERS)
    
    # Pre-calculate churn decisions
    churn_decisions = decide_churn(customers_df)
    
    # Generate all tables
    subscriptions_df = generate_subscriptions(customers_df, churn_decisions)
    transactions_df = generate_transactions(customers_df, subscriptions_df, churn_decisions)
    app_events_df = generate_app_events(customers_df, subscriptions_df, churn_decisions)
    support_tickets_df = generate_support_tickets(customers_df, subscriptions_df, churn_decisions)
    churn_labels_df = generate_churn_labels(customers_df, subscriptions_df, churn_decisions)
    
    # Save all CSVs
    print("\n" + "=" * 60)
    print("SAVING CSV FILES")
    print("=" * 60)
    
    datasets = {
        'customers.csv': customers_df,
        'subscriptions.csv': subscriptions_df,
        'transactions.csv': transactions_df,
        'app_events_daily.csv': app_events_df,
        'support_tickets.csv': support_tickets_df,
        'churn_labels.csv': churn_labels_df
    }
    
    for filename, df in datasets.items():
        filepath = raw_data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved: {filename} ({len(df):,} rows)")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Number of customers: {len(customers_df):,}")
    print(f"Churn rate: {churn_labels_df['churn'].mean() * 100:.2f}%")
    print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
    
    print("\n" + "-" * 40)
    print("Row counts:")
    for filename, df in datasets.items():
        print(f"  {filename}: {len(df):,} rows")
    
    print("\n" + "-" * 40)
    print("Data previews:")
    for filename, df in datasets.items():
        print(f"\n>>> {filename}:")
        print(df.head(3).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
