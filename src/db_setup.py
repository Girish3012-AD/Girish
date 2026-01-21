"""
db_setup.py
===========
SQLite Database Schema Setup for Customer Churn Project.

Creates tables with indexes for optimized joins on customer_id.

Tables:
- customers
- subscriptions
- transactions
- app_events_daily
- support_tickets
- churn_labels

Author: Senior Data Engineer + Data Scientist
"""

import sqlite3
from pathlib import Path

# Import configuration
try:
    from config import DB_PATH
except ImportError:
    from src.config import DB_PATH


# =============================================================================
# TABLE SCHEMAS
# =============================================================================

SCHEMA_CUSTOMERS = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    age INTEGER,
    gender TEXT,
    location TEXT,
    signup_date TEXT,
    device_type TEXT,
    acquisition_channel TEXT
);
"""

SCHEMA_SUBSCRIPTIONS = """
CREATE TABLE IF NOT EXISTS subscriptions (
    customer_id TEXT PRIMARY KEY,
    plan_type TEXT,
    monthly_price REAL,
    start_date TEXT,
    end_date TEXT,
    is_active INTEGER,
    auto_renew INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

SCHEMA_TRANSACTIONS = """
CREATE TABLE IF NOT EXISTS transactions (
    txn_id TEXT PRIMARY KEY,
    customer_id TEXT,
    txn_date TEXT,
    amount REAL,
    payment_status TEXT,
    payment_method TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

SCHEMA_APP_EVENTS_DAILY = """
CREATE TABLE IF NOT EXISTS app_events_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,
    event_date TEXT,
    sessions INTEGER,
    avg_session_minutes REAL,
    crashes INTEGER,
    content_category TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

SCHEMA_SUPPORT_TICKETS = """
CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id TEXT PRIMARY KEY,
    customer_id TEXT,
    ticket_date TEXT,
    issue_type TEXT,
    resolution_time_hours REAL,
    is_resolved INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

SCHEMA_CHURN_LABELS = """
CREATE TABLE IF NOT EXISTS churn_labels (
    customer_id TEXT PRIMARY KEY,
    churn INTEGER,
    churn_date TEXT,
    churn_reason TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

# =============================================================================
# INDEX DEFINITIONS (for faster joins)
# =============================================================================

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_subscriptions_customer ON subscriptions(customer_id);",
    "CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions(customer_id);",
    "CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(txn_date);",
    "CREATE INDEX IF NOT EXISTS idx_app_events_customer ON app_events_daily(customer_id);",
    "CREATE INDEX IF NOT EXISTS idx_app_events_date ON app_events_daily(event_date);",
    "CREATE INDEX IF NOT EXISTS idx_support_tickets_customer ON support_tickets(customer_id);",
    "CREATE INDEX IF NOT EXISTS idx_support_tickets_date ON support_tickets(ticket_date);",
    "CREATE INDEX IF NOT EXISTS idx_churn_labels_customer ON churn_labels(customer_id);",
]

# All schemas in order
ALL_SCHEMAS = [
    ("customers", SCHEMA_CUSTOMERS),
    ("subscriptions", SCHEMA_SUBSCRIPTIONS),
    ("transactions", SCHEMA_TRANSACTIONS),
    ("app_events_daily", SCHEMA_APP_EVENTS_DAILY),
    ("support_tickets", SCHEMA_SUPPORT_TICKETS),
    ("churn_labels", SCHEMA_CHURN_LABELS),
]


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def create_database(db_path: Path = None) -> None:
    """
    Create SQLite database with all required tables and indexes.
    
    Args:
        db_path: Path to database file. Defaults to config DB_PATH.
    """
    if db_path is None:
        db_path = DB_PATH
    
    print(f"Creating database at: {db_path}")
    
    # Connect to database (creates file if not exists)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Create each table
    print("\nCreating tables:")
    for table_name, schema in ALL_SCHEMAS:
        print(f"  ✓ {table_name}")
        cursor.execute(schema)
    
    # Create indexes for faster joins
    print("\nCreating indexes:")
    for index_sql in INDEXES:
        cursor.execute(index_sql)
        # Extract index name from SQL
        idx_name = index_sql.split("EXISTS ")[1].split(" ON")[0]
        print(f"  ✓ {idx_name}")
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print("\n✓ Database setup complete!")


def drop_all_tables(db_path: Path = None) -> None:
    """
    Drop all tables from the database.
    
    Args:
        db_path: Path to database file. Defaults to config DB_PATH.
    """
    if db_path is None:
        db_path = DB_PATH
    
    if not db_path.exists():
        print("Database does not exist. Nothing to drop.")
        return
    
    print(f"Dropping all tables from: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop tables in reverse order (due to foreign keys)
    for table_name, _ in reversed(ALL_SCHEMAS):
        print(f"  Dropping: {table_name}")
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
    
    conn.commit()
    conn.close()
    
    print("✓ All tables dropped!")


def reset_database(db_path: Path = None) -> None:
    """
    Reset database by dropping and recreating all tables.
    
    Args:
        db_path: Path to database file. Defaults to config DB_PATH.
    """
    if db_path is None:
        db_path = DB_PATH
    
    print("=" * 60)
    print("RESETTING DATABASE")
    print("=" * 60)
    
    drop_all_tables(db_path)
    create_database(db_path)


def verify_database(db_path: Path = None) -> None:
    """
    Verify database exists and print table info.
    
    Args:
        db_path: Path to database file. Defaults to config DB_PATH.
    """
    if db_path is None:
        db_path = DB_PATH
    
    if not db_path.exists():
        print(f"Database not found at: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = cursor.fetchall()
    
    print(f"\nDatabase: {db_path}")
    print(f"Tables found: {len(tables)}")
    
    for (table_name,) in tables:
        if table_name.startswith('sqlite_'):
            continue
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"  - {table_name}: {count:,} rows")
    
    conn.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to setup database."""
    print("=" * 60)
    print("CHURN DATABASE SETUP")
    print("=" * 60)
    
    # Reset and create fresh database
    reset_database()
    
    # Verify
    verify_database()
    
    print("\n" + "=" * 60)
    print("DATABASE SETUP COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
