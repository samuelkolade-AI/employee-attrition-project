"""
Airflow DAG for Employee Attrition Data Pipeline

This DAG uses BashOperator to run scripts directly.
No imports from the project needed - avoids all path issues.

SETUP:
1. Update PROJECT_ROOT below to your project path
2. Update VENV_ACTIVATE to point to your venv activate script
3. Copy this file to your Airflow dags folder
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# ============================================================
# UPDATE THESE TWO PATHS
# Run 'pwd' in your project root to get PROJECT_ROOT
# ============================================================
PROJECT_ROOT = "/home/kolade_ubuntu/employee-attrition-project"
VENV_ACTIVATE = f"{PROJECT_ROOT}/venv/bin/activate"

# ============================================================
# Default arguments
# ============================================================
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ============================================================
# Define DAG
# ============================================================
dag = DAG(
    'employee_attrition_pipeline',
    default_args=default_args,
    description='Daily ETL pipeline for employee attrition prediction',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    tags=['attrition', 'hr', 'ml'],
)

# ============================================================
# Task 1: Load Raw Data
# ============================================================
task_load_data = BashOperator(
    task_id='load_raw_data',
    bash_command=f"""
        source {VENV_ACTIVATE} && \
        cd {PROJECT_ROOT} && \
        python src/data/load_to_postgres.py
    """,
    dag=dag,
)

# ============================================================
# Task 2: Validate Data
# ============================================================
task_validate = BashOperator(
    task_id='validate_data',
    bash_command=f"""
        source {VENV_ACTIVATE} && \
        cd {PROJECT_ROOT} && \
        python src/data/validate_data.py
    """,
    dag=dag,
)

# ============================================================
# Task 3: Feature Engineering
# ============================================================
task_feature_eng = BashOperator(
    task_id='engineer_features',
    bash_command=f"""
        source {VENV_ACTIVATE} && \
        cd {PROJECT_ROOT} && \
        python src/features/build_features.py
    """,
    dag=dag,
)

# ============================================================
# Task 4: Train Model
# ============================================================
task_train = BashOperator(
    task_id='train_model',
    bash_command=f"""
        source {VENV_ACTIVATE} && \
        cd {PROJECT_ROOT} && \
        python src/models/train_model.py
    """,
    dag=dag,
)

# ============================================================
# Task 5: Analyze Features
# ============================================================
task_analyze = BashOperator(
    task_id='analyze_features',
    bash_command=f"""
        source {VENV_ACTIVATE} && \
        cd {PROJECT_ROOT} && \
        python src/models/analyze_features.py
    """,
    dag=dag,
)

# ============================================================
# Task Dependencies (execution order)
# ============================================================
# Flow: Load → Validate → Features → Train → Analyze
task_load_data >> task_validate >> task_feature_eng >> task_train >> task_analyze
