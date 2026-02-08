"""Load raw employee data into PostgreSQL."""

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from pathlib import Path
import sys
import re


def clean_column_name(col):
    """Convert any column name to PostgreSQL-friendly snake_case."""
    # Remove any suffix like _m999
    col = re.sub(r'_m\d+$', '', col)

    # Convert to snake_case
    col = re.sub(r'(?<!^)(?=[A-Z])', '_', col)
    col = col.lower()
    col = re.sub(r'__+', '_', col)
    col = col.strip('_')

    return col


def load_data_to_postgres():
    """Load CSV data into PostgreSQL raw_employees table."""

    DATABASE_URL = "postgresql://attrition_user:attrition_DBpass001@localhost:5432/employee_attrition"

    try:
        engine = create_engine(DATABASE_URL)
        print("✅ Connected to PostgreSQL")

        # Load CSV
        csv_path = Path('data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv')
        if not csv_path.exists():
            print(f"❌ Error: CSV file not found at {csv_path}")
            return False

        df = pd.read_csv(csv_path)
        print(f"📂 Loaded {len(df)} records from CSV")

        # Auto-clean all column names
        print("🔧 Cleaning column names...")
        df.columns = [clean_column_name(col) for col in df.columns]

        # Handle employee ID
        if 'employee_number' in df.columns:
            df = df.rename(columns={'employee_number': 'employee_id'})
            df['employee_id'] = 'EMP' + df['employee_id'].astype(str)
        elif 'employee_id' not in df.columns:
            df['employee_id'] = 'EMP' + (df.index + 1).astype(str)

        # Drop useless columns
        cols_to_drop = ['employee_count', 'over18', 'standard_hours']
        df = df.drop(
            columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

        print(f"📊 Final dataset: {len(df)} rows × {len(df.columns)} columns")

        # Check if table exists
        inspector = inspect(engine)
        table_exists = 'raw_employees' in inspector.get_table_names()

        if table_exists:
            print("⚠️  Table exists. Clearing old data...")
            with engine.connect() as conn:
                conn.execute(text("TRUNCATE TABLE raw_employees CASCADE"))
                conn.commit()
            print("✅ Old data cleared")

        # Load data - SIMPLE METHOD (no method='multi')
        print("📤 Loading data to PostgreSQL...")
        df.to_sql(
            'raw_employees',
            engine,
            if_exists='append' if table_exists else 'replace',
            index=False,
            chunksize=500
        )

        print(f"✅ Loaded {len(df)} records into PostgreSQL")

        # Verify
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) as count FROM raw_employees"))
            count = result.fetchone()[0]
            print(f"✅ Verification: {count} records in database")

        # Sample data
        sample = pd.read_sql("SELECT * FROM raw_employees LIMIT 3", engine)
        print("\n📊 Sample Data:")
        cols_to_show = ['employee_id', 'age',
                        'department', 'attrition', 'monthly_income']
        available_cols = [c for c in cols_to_show if c in sample.columns]
        if available_cols:
            print(sample[available_cols].to_string(index=False))

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = load_data_to_postgres()
    sys.exit(0 if success else 1)
