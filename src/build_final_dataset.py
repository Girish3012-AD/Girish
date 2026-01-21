"""
build_final_dataset.py
======================
Build ML-ready dataset from SQLite database using SQL aggregations.

Output Columns:
- customer_id, age, gender, location, device_type, acquisition_channel
- plan_type, monthly_price, auto_renew
- total_sessions_30d, avg_session_minutes_30d, total_crashes_30d
- failed_payments_30d, total_amount_success_30d
- support_tickets_30d, avg_resolution_time_30d
- churn

Author: Senior Data Engineer + Data Scientist
"""

import sqlite3
import pandas as pd
from pathlib import Path

# Import configuration
try:
    from config import DB_PATH, OUTPUT_DIR, FINAL_DATASET_FILENAME, FINAL_DATASET_COLUMNS
    from build_features_sql import get_ml_dataset_query, QUERY_ALL_MAX_DATES
except ImportError:
    from src.config import DB_PATH, OUTPUT_DIR, FINAL_DATASET_FILENAME, FINAL_DATASET_COLUMNS
    from src.build_features_sql import get_ml_dataset_query, QUERY_ALL_MAX_DATES


# =============================================================================
# DATASET BUILDING FUNCTIONS
# =============================================================================

def get_max_dates(conn: sqlite3.Connection) -> tuple:
    """
    Get maximum dates from each table for rolling window calculations.
    
    Args:
        conn: SQLite connection
    
    Returns:
        Tuple of (max_event_date, max_txn_date, max_ticket_date)
    """
    cursor = conn.cursor()
    cursor.execute(QUERY_ALL_MAX_DATES)
    result = cursor.fetchone()
    
    # Handle None values with default fallback
    max_event_date = result[0] or '2025-12-31'
    max_txn_date = result[1] or '2025-12-31'
    max_ticket_date = result[2] or '2025-12-31'
    
    return max_event_date, max_txn_date, max_ticket_date


def build_ml_dataset(db_path: Path = None) -> pd.DataFrame:
    """
    Build ML-ready dataset from SQLite database.
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        DataFrame with ML-ready features and churn label
    """
    if db_path is None:
        db_path = DB_PATH
    
    print(f"Building ML dataset from: {db_path}")
    
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Please run db_setup.py and load_to_sqlite.py first."
        )
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Get max dates for rolling window
    max_event_date, max_txn_date, max_ticket_date = get_max_dates(conn)
    
    print(f"\n  Max event date: {max_event_date}")
    print(f"  Max txn date: {max_txn_date}")
    print(f"  Max ticket date: {max_ticket_date}")
    print(f"  Aggregation window: 30 days")
    
    # Build and execute query
    query = get_ml_dataset_query(max_event_date, max_txn_date, max_ticket_date)
    df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    # Ensure column order matches expected columns
    df = df[FINAL_DATASET_COLUMNS]
    
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset for ML readiness.
    
    - Remove any remaining object date columns
    - Ensure numeric columns are proper types
    - Fill any remaining NaN values
    
    Args:
        df: Input DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Define column types
    numeric_cols = [
        'age', 'monthly_price', 'auto_renew',
        'total_sessions_30d', 'avg_session_minutes_30d', 'total_crashes_30d',
        'failed_payments_30d', 'total_amount_success_30d',
        'support_tickets_30d', 'avg_resolution_time_30d', 'churn'
    ]
    
    categorical_cols = ['gender', 'location', 'device_type', 'acquisition_channel', 'plan_type']
    
    # Convert numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Ensure categorical columns are strings
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('None', 'Unknown')
    
    # Convert churn to integer
    df['churn'] = df['churn'].astype(int)
    
    return df


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save dataset to CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path to output CSV file
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✓ Dataset saved to: {output_path}")


def print_summary(df: pd.DataFrame) -> None:
    """
    Print comprehensive dataset summary.
    
    Args:
        df: DataFrame to summarize
    """
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    
    # Shape
    print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Column list
    print(f"\n📋 Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Churn distribution
    churn_counts = df['churn'].value_counts()
    churn_rate = df['churn'].mean() * 100
    
    print(f"\n🎯 Churn Distribution:")
    print(f"   - Not Churned (0): {churn_counts.get(0, 0):,} ({100 - churn_rate:.2f}%)")
    print(f"   - Churned (1): {churn_counts.get(1, 0):,} ({churn_rate:.2f}%)")
    
    # Feature statistics
    print(f"\n📈 Feature Statistics:")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in numeric_cols[:10]:  # Limit to first 10
        print(f"   - {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
    
    # Missing values check
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\n🔍 Missing Values: {total_missing}")
    if total_missing > 0:
        for col, count in missing[missing > 0].items():
            print(f"   - {col}: {count:,}")
    
    # Data types
    print(f"\n🏷️ Data Types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"   - {dtype}: {count} columns")
    
    # Sample data
    print(f"\n📝 Sample Data (first 5 rows):")
    print(df.head().to_string(index=False))


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to build and save ML dataset."""
    print("=" * 60)
    print("ML DATASET BUILDER")
    print("=" * 60)
    
    # Build dataset
    df = build_ml_dataset()
    
    # Clean dataset
    df = clean_dataset(df)
    
    # Print summary
    print_summary(df)
    
    # Save dataset
    output_path = OUTPUT_DIR / FINAL_DATASET_FILENAME
    save_dataset(df, output_path)
    
    print("\n" + "=" * 60)
    print("✅ ML DATASET BUILD COMPLETE!")
    print("=" * 60)
    print(f"\n   📁 Output: {output_path}")
    print(f"   📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   🎯 Churn Rate: {df['churn'].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
