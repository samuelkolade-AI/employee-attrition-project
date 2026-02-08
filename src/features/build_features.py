"""
Feature Engineering for Employee Attrition Prediction
Creates ML-ready features from raw employee data
"""

import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import sys

DATABASE_URL = "postgresql://attrition_user:attrition_DBpass001@localhost:5432/employee_attrition"


def engineer_features():
    """
    Create ML-ready features from raw employee data.

    Features created:
    - Tenure-based features
    - Satisfaction scores
    - Compensation ratios
    - Categorical encodings
    - Risk flags
    """

    print("🔧 Starting feature engineering...")

    try:
        # Connect to database
        engine = create_engine(DATABASE_URL)

        # Load raw data
        df = pd.read_sql("SELECT * FROM raw_employees", engine)
        print(f"📊 Loaded {len(df)} employee records")

        # Create feature dataframe
        features = pd.DataFrame()
        features['employee_id'] = df['employee_id']

        # === NUMERICAL FEATURES (Direct from raw data) ===
        features['age'] = df['age']
        features['tenure_years'] = df['years_at_company']
        features['promotion_gap_years'] = df['years_since_last_promotion']
        features['monthly_income'] = df['monthly_income']
        features['distance_from_home'] = df['distance_from_home']
        features['total_working_years'] = df['total_working_years']
        features['years_in_current_role'] = df['years_in_current_role']
        features['years_with_curr_manager'] = df['years_with_curr_manager']
        features['num_companies_worked'] = df['num_companies_worked']
        features['percent_salary_hike'] = df['percent_salary_hike']
        features['training_times_last_year'] = df['training_times_last_year']
        features['job_level'] = df['job_level']
        features['stock_option_level'] = df['stock_option_level']

        # === ENGINEERED FEATURES ===

        # 1. Tenure ratios (avoid division by zero)
        features['role_tenure_ratio'] = df['years_in_current_role'] / \
            (df['years_at_company'] + 1)
        features['promotion_delay_ratio'] = df['years_since_last_promotion'] / \
            (df['years_at_company'] + 1)
        features['manager_stability'] = df['years_with_curr_manager'] / \
            (df['years_in_current_role'] + 1)

        # 2. Compensation features
        # Salary per year of experience
        features['income_per_experience_year'] = df['monthly_income'] / \
            (df['total_working_years'] + 1)

        # Estimate market salary based on tenure groups
        tenure_bins = pd.cut(df['total_working_years'],
                             bins=[0, 5, 10, 15, 100])
        avg_income_by_tenure = df.groupby(
            tenure_bins)['monthly_income'].transform('mean')
        features['salary_to_market_ratio'] = df['monthly_income'] / \
            (avg_income_by_tenure + 1)

        # Recent salary growth indicator
        features['low_salary_hike_flag'] = (
            df['percent_salary_hike'] < 13).astype(int)

        # 3. Satisfaction composite scores
        features['satisfaction_score'] = (
            df['job_satisfaction'] +
            df['environment_satisfaction'] +
            df['relationship_satisfaction']
        ) / 3

        features['engagement_score'] = (
            df['job_involvement'] +
            df['work_life_balance']
        ) / 2

        # Individual satisfaction flags
        features['low_job_satisfaction'] = (
            df['job_satisfaction'] <= 2).astype(int)
        features['low_environment_satisfaction'] = (
            df['environment_satisfaction'] <= 2).astype(int)
        features['poor_work_life_balance'] = (
            df['work_life_balance'] <= 2).astype(int)
        features['low_relationship_satisfaction'] = (
            df['relationship_satisfaction'] <= 2).astype(int)

        # 4. Career progression features
        features['stagnant_career'] = (
            (df['years_since_last_promotion'] > 5) &
            (df['years_at_company'] > 5)
        ).astype(int)

        features['early_career'] = (df['total_working_years'] <= 3).astype(int)
        features['job_hopper'] = (df['num_companies_worked'] > 5).astype(int)
        features['long_tenure'] = (df['years_at_company'] > 10).astype(int)

        # 5. Work conditions
        features['overtime_flag'] = (df['over_time'] == 'Yes').astype(int)
        features['high_distance_flag'] = (
            df['distance_from_home'] > 20).astype(int)
        features['frequent_traveler'] = (
            df['business_travel'] == 'Travel_Frequently').astype(int)
        features['non_traveler'] = (
            df['business_travel'] == 'Non-Travel').astype(int)

        # 6. Demographics
        features['is_young'] = (df['age'] < 30).astype(int)
        features['is_mid_career'] = (
            (df['age'] >= 30) & (df['age'] < 45)).astype(int)
        features['is_senior'] = (df['age'] >= 45).astype(int)
        features['is_single'] = (df['marital_status'] == 'Single').astype(int)
        features['is_married'] = (
            df['marital_status'] == 'Married').astype(int)
        features['is_male'] = (df['gender'] == 'Male').astype(int)

        # 7. Role-based features
        features['is_sales'] = (df['department'] == 'Sales').astype(int)
        features['is_rd'] = (
            df['department'] == 'Research & Development').astype(int)
        features['is_hr'] = (df['department'] == 'Human Resources').astype(int)
        features['low_job_level'] = (df['job_level'] <= 2).astype(int)
        features['high_job_level'] = (df['job_level'] >= 4).astype(int)

        # 8. Training and development
        features['low_training'] = (
            df['training_times_last_year'] <= 1).astype(int)
        features['high_training'] = (
            df['training_times_last_year'] >= 4).astype(int)

        # 9. Stock options (retention tool)
        features['has_stock_options'] = (
            df['stock_option_level'] > 0).astype(int)
        features['high_stock_options'] = (
            df['stock_option_level'] >= 2).astype(int)

        # 10. Performance indicators
        features['high_performer'] = (
            df['performance_rating'] == 4).astype(int)
        features['job_involvement_score'] = df['job_involvement']

        # 11. Education features
        features['education_level'] = df['education']
        features['life_sciences_field'] = (
            df['education_field'] == 'Life Sciences').astype(int)
        features['medical_field'] = (
            df['education_field'] == 'Medical').astype(int)
        features['technical_field'] = (
            df['education_field'] == 'Technical Degree').astype(int)

        # 12. Combined risk indicators
        features['high_risk_profile'] = (
            (features['overtime_flag'] == 1) &
            (features['low_job_satisfaction'] == 1) &
            (features['low_salary_hike_flag'] == 1)
        ).astype(int)

        features['retention_risk_score'] = (
            features['overtime_flag'] +
            features['low_job_satisfaction'] +
            features['stagnant_career'] +
            features['low_salary_hike_flag'] +
            features['poor_work_life_balance']
        )

        # === TARGET VARIABLE ===
        features['attrition_label'] = (df['attrition'] == 'Yes').astype(int)

        # === HANDLE MISSING/INFINITE VALUES ===
        # Replace infinite values with NaN
        features = features.replace([np.inf, -np.inf], np.nan)

        # Fill NaN with 0
        features = features.fillna(0)

        # === DATA TYPE VALIDATION ===
        # Ensure all features are numeric
        non_numeric = features.select_dtypes(
            exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"⚠️  Warning: Non-numeric columns found: {non_numeric}")
            # Convert to numeric if possible
            for col in non_numeric:
                if col != 'employee_id':
                    features[col] = pd.to_numeric(
                        features[col], errors='coerce').fillna(0)

        # === SAVE TO DATABASE ===
        print(f"\n💾 Saving features to database...")
        features.to_sql(
            'ml_features',
            engine,
            if_exists='replace',
            index=False,
            method='multi',
            chunksize=1000
        )

        print(
            f"✅ Created {len(features.columns) - 2} features for {len(features)} employees")
        print(f"✅ Saved to 'ml_features' table in PostgreSQL")

        # === FEATURE SUMMARY ===
        print("\n" + "=" * 80)
        print("📊 FEATURE ENGINEERING SUMMARY")
        print("=" * 80)

        # Exclude employee_id and target
        print(f"\nTotal Features Created: {len(features.columns) - 2}")
        print(f"Total Employees: {len(features)}")

        # Show feature categories
        print("\n📋 Feature Categories:")
        print(f"  • Numerical Features: 13")
        print(f"  • Tenure Ratios: 3")
        print(f"  • Compensation Features: 3")
        print(f"  • Satisfaction Scores: 6")
        print(f"  • Career Progression: 4")
        print(f"  • Work Conditions: 3")
        print(f"  • Demographics: 6")
        print(f"  • Role-based: 5")
        print(f"  • Training: 2")
        print(f"  • Stock Options: 2")
        print(f"  • Performance: 2")
        print(f"  • Education: 4")
        print(f"  • Risk Indicators: 2")

        # Check attrition distribution
        attrition_rate = features['attrition_label'].mean()
        attrition_count = features['attrition_label'].sum()
        print(f"\n📈 Target Variable Distribution:")
        print(f"  • Attrition Rate: {attrition_rate:.2%}")
        print(f"  • Employees Who Left: {attrition_count}")
        print(f"  • Employees Retained: {len(features) - attrition_count}")

        # Show sample statistics for key features
        print("\n📊 Sample Feature Statistics:")
        key_features = [
            'age', 'tenure_years', 'monthly_income', 'satisfaction_score',
            'engagement_score', 'retention_risk_score'
        ]
        print(features[key_features].describe().round(2))

       # Show correlations with attrition
        print("\n🔗 Top 10 Features Correlated with Attrition:")
        # Only use numeric columns for correlation
        numeric_features = features.select_dtypes(include=[np.number])
        correlations = numeric_features.corr(
        )['attrition_label'].sort_values(ascending=False)
        # Top 11 (includes attrition itself)
        print(correlations.head(11).to_string())

        print("\n" + "=" * 80)
        print("✅ FEATURE ENGINEERING COMPLETE!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = engineer_features()
    sys.exit(0 if success else 1)
