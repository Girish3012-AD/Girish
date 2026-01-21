"""
config.py
=========
Centralized configuration for Customer Churn Project.

Contains all paths, settings, and constants used across the project.
Uses Path(__file__).resolve() pattern to avoid hardcoded paths.

Author: Senior Data Engineer + Data Scientist
"""

from pathlib import Path


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

# Database path
DB_PATH = PROJECT_ROOT / 'churn.db'


# =============================================================================
# CSV FILE CONFIGURATION
# =============================================================================

# Mapping of CSV filenames to SQLite table names
CSV_TABLE_MAPPING = {
    'customers.csv': 'customers',
    'subscriptions.csv': 'subscriptions',
    'transactions.csv': 'transactions',
    'app_events_daily.csv': 'app_events_daily',
    'support_tickets.csv': 'support_tickets',
    'churn_labels.csv': 'churn_labels',
}


# =============================================================================
# FINAL DATASET CONFIGURATION
# =============================================================================

# Output filename for final ML dataset
FINAL_DATASET_FILENAME = 'final_churn_dataset.csv'

# Expected columns in final dataset (in order)
FINAL_DATASET_COLUMNS = [
    'customer_id',
    'age',
    'gender',
    'location',
    'device_type',
    'acquisition_channel',
    'plan_type',
    'monthly_price',
    'auto_renew',
    'total_sessions_30d',
    'avg_session_minutes_30d',
    'total_crashes_30d',
    'failed_payments_30d',
    'total_amount_success_30d',
    'support_tickets_30d',
    'avg_resolution_time_30d',
    'churn',
]


# =============================================================================
# AGGREGATION WINDOW CONFIGURATION
# =============================================================================

# Number of days for rolling window aggregations
AGGREGATION_WINDOW_DAYS = 30


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_directories_exist():
    """Create all required directories if they don't exist."""
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def print_config():
    """Print current configuration for debugging."""
    print("=" * 50)
    print("PROJECT CONFIGURATION")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Data Dir: {RAW_DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Database Path: {DB_PATH}")
    print(f"Aggregation Window: {AGGREGATION_WINDOW_DAYS} days")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
