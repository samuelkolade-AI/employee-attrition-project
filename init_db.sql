-- Initialize database schema for employee attrition prediction

-- Drop tables if they exist (for clean restart)
DROP TABLE IF EXISTS attrition_predictions CASCADE;
DROP TABLE IF EXISTS ml_features CASCADE;
DROP TABLE IF EXISTS raw_employees CASCADE;

-- Raw employee data table (matches CSV structure exactly)
CREATE TABLE IF NOT EXISTS raw_employees (
    employee_id VARCHAR(50) PRIMARY KEY,
    age INTEGER,
    attrition VARCHAR(10),
    business_travel VARCHAR(50),
    daily_rate INTEGER,
    department VARCHAR(100),
    distance_from_home INTEGER,
    education INTEGER,
    education_field VARCHAR(100),
    environment_satisfaction INTEGER,
    gender VARCHAR(20),
    hourly_rate INTEGER,
    job_involvement INTEGER,
    job_level INTEGER,
    job_role VARCHAR(100),
    job_satisfaction INTEGER,
    marital_status VARCHAR(50),
    monthly_income INTEGER,
    monthly_rate INTEGER,
    num_companies_worked INTEGER,
    over_time VARCHAR(10),
    percent_salary_hike INTEGER,
    performance_rating INTEGER,
    relationship_satisfaction INTEGER,
    stock_option_level INTEGER,
    total_working_years INTEGER,
    training_times_last_year INTEGER,
    work_life_balance INTEGER,
    years_at_company INTEGER,
    years_in_current_role INTEGER,
    years_since_last_promotion INTEGER,
    years_with_curr_manager INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature store table (will be populated by feature engineering)
CREATE TABLE IF NOT EXISTS ml_features (
    employee_id VARCHAR(50) PRIMARY KEY,
    -- Features will be created dynamically by the feature engineering script
    -- This table will be dropped and recreated during feature engineering
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions table
CREATE TABLE IF NOT EXISTS attrition_predictions (
    prediction_id SERIAL PRIMARY KEY,
    employee_id VARCHAR(50),
    prediction_date DATE,
    attrition_probability FLOAT,
    risk_level VARCHAR(20),
    top_risk_factor_1 VARCHAR(100),
    top_risk_factor_2 VARCHAR(100),
    top_risk_factor_3 VARCHAR(100),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_attrition ON raw_employees(attrition);
CREATE INDEX IF NOT EXISTS idx_department ON raw_employees(department);
CREATE INDEX IF NOT EXISTS idx_employee_id ON raw_employees(employee_id);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO attrition_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO attrition_user;

-- Display confirmation
DO $$
BEGIN
    RAISE NOTICE 'Database schema initialized successfully!';
    RAISE NOTICE 'Tables created: raw_employees, ml_features, attrition_predictions';
END $$;