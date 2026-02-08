"""
Data Quality Validation for ML Features
"""

import pandas as pd
from sqlalchemy import create_engine
import sys

# Update with your actual password
DATABASE_URL = "postgresql://attrition_user:attrition_DBpass001@localhost:5432/employee_attrition"


def validate_features():
    """Validate ML features meet quality standards."""

    print("🔍 Starting data quality validation...")

    try:
        engine = create_engine(DATABASE_URL)
        df = pd.read_sql("SELECT * FROM ml_features", engine)

        print(f"📊 Validating {len(df)} records with {len(df.columns)} columns")

        issues = []

        # Check 1: No missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            issues.append(
                f"Missing values found: {missing[missing > 0].to_dict()}")
        else:
            print("✅ No missing values")

        # Check 2: Attrition label is binary
        if 'attrition_label' in df.columns:
            unique_vals = set(df['attrition_label'].unique())
            if unique_vals != {0, 1} and unique_vals != {0.0, 1.0}:
                issues.append(
                    f"Attrition label has unexpected values: {unique_vals}")
            else:
                print("✅ Attrition label is binary")

        # Check 3: Reasonable value ranges
        if 'age' in df.columns:
            if (df['age'] < 18).any() or (df['age'] > 70).any():
                issues.append("Age values outside reasonable range (18-70)")
            else:
                print("✅ Age values in reasonable range")

        if 'tenure_years' in df.columns:
            if (df['tenure_years'] < 0).any():
                issues.append("Negative tenure values found")
            else:
                print("✅ Tenure values are non-negative")

        # Check 4: Feature count
        expected_features = 50  # At least 50 features
        if len(df.columns) < expected_features:
            issues.append(
                f"Expected at least {expected_features} features, got {len(df.columns)}")
        else:
            print(f"✅ Feature count OK ({len(df.columns)} columns)")

        # Check 5: No duplicate employee IDs
        if 'employee_id' in df.columns:
            if df['employee_id'].duplicated().any():
                issues.append("Duplicate employee IDs found")
            else:
                print("✅ No duplicate employee IDs")

        # Check 6: Class balance (attrition rate should be 5-30%)
        if 'attrition_label' in df.columns:
            attrition_rate = df['attrition_label'].mean()
            if attrition_rate < 0.05 or attrition_rate > 0.30:
                issues.append(f"Unusual attrition rate: {attrition_rate:.2%}")
            else:
                print(f"✅ Attrition rate is reasonable: {attrition_rate:.2%}")

        # Summary
        print("\n" + "=" * 80)
        if issues:
            print("❌ VALIDATION FAILED")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ ALL VALIDATIONS PASSED")
            print("=" * 80)
            return True

    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = validate_features()
    sys.exit(0 if success else 1)
