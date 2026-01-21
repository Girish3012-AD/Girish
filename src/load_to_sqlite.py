"""
load_to_sqlite.py
=================
Load CSV files from data/raw/ into SQLite database.

Replaces existing tables if they already exist.
Converts date fields safely using pandas.

Author: Senior Data Engineer + Data Scientist
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import configuration
try:
    from config import DB_PATH, RAW_DATA_DIR, CSV_TABLE_MAPPING
except ImportError:
    from src.config import DB_PATH, RAW_DATA_DIR, CSV_TABLE_MAPPING


# =============================================================================
# DATE COLUMNS FOR EACH TABLE
# =============================================================================

DATE_COLUMNS = {
    'customers': ['signup_date'],
    'subscriptions': ['start_date', 'end_date'],
    'transactions': ['txn_date'],
    'app_events_daily': ['event_date'],
    'support_tickets': ['ticket_date'],
    'churn_labels': ['churn_date'],
}


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def convert_dates(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Convert date columns to proper date format for SQLite.
    
    Args:
        df: DataFrame to process
        table_name: Name of the table (to look up date columns)
    
    Returns:
        DataFrame with converted date columns
    """
    df = df.copy()
    date_cols = DATE_COLUMNS.get(table_name, [])
    
    for col in date_cols:
        if col in df.columns:
            # Convert to datetime, then to string format YYYY-MM-DD
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].dt.strftime('%Y-%m-%d')
            # Replace 'NaT' string with None
            df[col] = df[col].replace('NaT', None)
    
    return df


def load_csv_to_sqlite(csv_path: Path, table_name: str, conn: sqlite3.Connection,
                       if_exists: str = 'replace') -> int:
    """
    Load a single CSV file into SQLite table.
    
    Args:
        csv_path: Path to CSV file
        table_name: Target table name in SQLite
        conn: SQLite connection object
        if_exists: How to handle existing table ('replace', 'append', 'fail')
    
    Returns:
        Number of rows loaded
    """
    if not csv_path.exists():
        print(f"  ⚠ WARNING: File not found - {csv_path.name}")
        return 0
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Convert date columns
    df = convert_dates(df, table_name)
    
    # Load into SQLite (replaces existing table)
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    
    return len(df)


def load_all_csvs(raw_data_dir: Path = None, db_path: Path = None) -> dict:
    """
    Load all CSV files from raw data directory into SQLite.
    
    Args:
        raw_data_dir: Path to raw data directory
        db_path: Path to SQLite database
    
    Returns:
        Dictionary mapping table names to row counts
    """
    if raw_data_dir is None:
        raw_data_dir = RAW_DATA_DIR
    
    if db_path is None:
        db_path = DB_PATH
    
    print(f"Loading data from: {raw_data_dir}")
    print(f"Target database: {db_path}")
    print("-" * 50)
    
    # Verify paths
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
    
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Please run db_setup.py first to create the database."
        )
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Track results
    results = {}
    
    # Load each CSV
    for csv_filename, table_name in CSV_TABLE_MAPPING.items():
        csv_path = raw_data_dir / csv_filename
        rows_loaded = load_csv_to_sqlite(csv_path, table_name, conn)
        results[table_name] = rows_loaded
        print(f"  ✓ Loaded {table_name}: {rows_loaded:,} rows")
    
    # Commit and close
    conn.commit()
    conn.close()
    
    return results


def verify_data_loaded(db_path: Path = None) -> None:
    """
    Verify data was loaded correctly by querying each table.
    
    Args:
        db_path: Path to SQLite database
    """
    if db_path is None:
        db_path = DB_PATH
    
    conn = sqlite3.connect(db_path)
    
    print("\nVerification - Sample data from each table:")
    print("=" * 60)
    
    for table_name in CSV_TABLE_MAPPING.values():
        try:
            query = f"SELECT * FROM {table_name} LIMIT 3;"
            df = pd.read_sql_query(query, conn)
            print(f"\n>>> {table_name} ({len(df)} sample rows):")
            print(df.to_string(index=False))
        except Exception as e:
            print(f"\n>>> {table_name}: Error - {e}")
    
    conn.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to load all data."""
    print("=" * 60)
    print("CHURN DATA LOADER")
    print("=" * 60)
    
    # Load all CSVs
    results = load_all_csvs()
    
    # Print summary
    print("\n" + "=" * 60)
    print("LOAD SUMMARY")
    print("=" * 60)
    total_rows = sum(results.values())
    print(f"Total rows loaded: {total_rows:,}")
    
    for table, rows in results.items():
        print(f"  - {table}: {rows:,} rows")
    
    # Verify
    verify_data_loaded()
    
    print("\n" + "=" * 60)
    print("DATA LOADING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
