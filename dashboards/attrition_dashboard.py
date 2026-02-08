"""
Interactive Dashboard for Employee Attrition Analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import requests

# Page config
st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="👔",
    layout="wide"
)

# Update with your password
DATABASE_URL = "postgresql://attrition_user:attrition_DBpass001@localhost:5432/employee_attrition"
API_URL = "http://localhost:8000"


@st.cache_data
def load_data():
    """Load employee data from PostgreSQL."""
    try:
        engine = create_engine(DATABASE_URL)
        raw_df = pd.read_sql("SELECT * FROM raw_employees", engine)
        features_df = pd.read_sql("SELECT * FROM ml_features", engine)
        return raw_df, features_df
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, None


def make_prediction(features):
    """Call API to make prediction."""
    try:
        response = requests.post(
            f"{API_URL}/predict", json=features, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.warning(
            f"Could not connect to prediction API. Make sure it's running on {API_URL}")
        return None


def main():
    st.title("👔 Employee Attrition Analytics Dashboard")
    st.markdown("### Predict and prevent employee turnover with AI")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "📊 Overview",
        "🔮 Predict Attrition",
        "📈 Feature Importance"
    ])

    # Load data
    raw_df, features_df = load_data()

    if raw_df is None:
        st.error("Failed to load data. Please check database connection.")
        st.info("Make sure PostgreSQL is running: `docker ps`")
        return

    if page == "📊 Overview":
        show_overview(raw_df, features_df)
    elif page == "🔮 Predict Attrition":
        show_prediction_tool()
    elif page == "📈 Feature Importance":
        show_feature_importance()


def show_overview(raw_df, features_df):
    """Display overview metrics and charts."""
    st.header("📊 Company Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    total_employees = len(raw_df)
    attrition_count = (raw_df['attrition'] == 'Yes').sum()
    attrition_rate = attrition_count / total_employees
    avg_tenure = raw_df['years_at_company'].mean()

    with col1:
        st.metric("Total Employees", f"{total_employees:,}")
    with col2:
        st.metric("Attrition Rate", f"{attrition_rate:.1%}",
                  delta=f"-{attrition_count} employees", delta_color="inverse")
    with col3:
        st.metric("Average Tenure", f"{avg_tenure:.1f} years")
    with col4:
        avg_income = raw_df['monthly_income'].mean()
        st.metric("Avg Monthly Income", f"${avg_income:,.0f}")

    st.markdown("---")

    # Charts Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition by Department")
        dept_data = pd.crosstab(
            raw_df['department'], raw_df['attrition'], normalize='index') * 100
        fig = px.bar(
            dept_data.reset_index(),
            x='department',
            y='Yes',
            labels={'Yes': 'Attrition Rate (%)', 'department': 'Department'},
            color='Yes',
            color_continuous_scale='Reds',
            title="Which departments are losing talent?"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Age Distribution by Attrition")
        fig = px.histogram(
            raw_df,
            x='age',
            color='attrition',
            nbins=20,
            labels={'attrition': 'Attrition'},
            color_discrete_map={'Yes': '#ff4b4b', 'No': '#00cc00'},
            title="Younger employees leave more frequently"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Charts Row 2
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Income Distribution")
        fig = px.box(
            raw_df,
            x='attrition',
            y='monthly_income',
            color='attrition',
            labels={'attrition': 'Attrition',
                    'monthly_income': 'Monthly Income ($)'},
            color_discrete_map={'Yes': '#ff4b4b', 'No': '#00cc00'},
            title="Lower paid employees more likely to leave"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Overtime Impact")
        overtime_data = pd.crosstab(
            raw_df['over_time'], raw_df['attrition'], normalize='index') * 100
        fig = px.bar(
            overtime_data.reset_index(),
            x='over_time',
            y='Yes',
            labels={'Yes': 'Attrition Rate (%)', 'over_time': 'Overtime'},
            color='Yes',
            color_continuous_scale='Reds',
            title="Overtime significantly increases attrition risk"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Risk Distribution
    st.markdown("---")
    st.subheader("🎯 Risk Distribution")

    if features_df is not None and 'retention_risk_score' in features_df.columns:
        # Create risk categories based on retention_risk_score
        features_df['risk_category'] = pd.cut(
            features_df['retention_risk_score'],
            bins=[0, 1, 2, 5],
            labels=['Low', 'Medium', 'High']
        )

        risk_dist = features_df['risk_category'].value_counts()

        col1, col2, col3 = st.columns(3)

        with col1:
            low_risk = risk_dist.get('Low', 0)
            st.metric("🟢 Low Risk", f"{low_risk:,} employees")
        with col2:
            med_risk = risk_dist.get('Medium', 0)
            st.metric("🟡 Medium Risk", f"{med_risk:,} employees")
        with col3:
            high_risk = risk_dist.get('High', 0)
            st.metric("🔴 High Risk", f"{high_risk:,} employees")


def show_prediction_tool():
    """Interactive prediction tool."""
    st.header("🔮 Predict Employee Attrition Risk")
    st.markdown(
        "Enter employee information to predict their likelihood of leaving")

    # Check if API is running
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=2)
        if health_response.status_code == 200:
            st.success("✅ Prediction API is online")
        else:
            st.warning("⚠️ Prediction API is not responding correctly")
    except:
        st.error(
            "❌ Prediction API is offline. Start it with: `python src/api/main.py`")
        return

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Information")
        age = st.number_input("Age", min_value=18,
                              max_value=70, value=35, step=1)
        tenure = st.number_input(
            "Years at Company", min_value=0, max_value=40, value=5, step=1)
        income = st.number_input(
            "Monthly Income ($)", min_value=1000, max_value=50000, value=5000, step=500)

        st.subheader("Work Conditions")
        overtime = st.selectbox("Works Overtime?", ["No", "Yes"])
        satisfaction = st.slider("Job Satisfaction (1-4)", 1.0, 4.0, 3.0, 0.5)

    with col2:
        st.subheader("Career Progression")
        promotion_gap = st.number_input(
            "Years Since Last Promotion", min_value=0, max_value=20, value=2, step=1)
        job_level = st.slider("Job Level (1-5)", 1, 5, 2)

        st.subheader("Engagement")
        engagement = st.slider("Engagement Score (1-4)", 1.0, 4.0, 3.0, 0.5)
        training = st.number_input(
            "Training Times Last Year", min_value=0, max_value=10, value=2, step=1)

    st.markdown("---")

    if st.button("🎯 Predict Attrition Risk", type="primary", use_container_width=True):
        # Prepare features
        features = {
            "age": age,
            "tenure_years": tenure,
            "monthly_income": float(income),
            "satisfaction_score": float(satisfaction),
            "overtime_flag": 1 if overtime == "Yes" else 0,
            "promotion_gap_years": promotion_gap,
            "job_level": job_level,
            "engagement_score": float(engagement),
            "training_times_last_year": training,
            "low_job_satisfaction": 1 if satisfaction <= 2 else 0,
            "stagnant_career": 1 if (promotion_gap > 5 and tenure > 5) else 0,
            "is_young": 1 if age < 30 else 0,
        }

        # Make prediction
        with st.spinner("Analyzing employee profile..."):
            result = make_prediction(features)

        if result:
            st.markdown("---")
            st.subheader("📊 Prediction Results")

            prob = result['attrition_probability']
            risk_level = result['risk_level']

            # Display gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Attrition Risk Score", 'font': {'size': 24}},
                number={'suffix': "%", 'font': {'size': 40}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ))

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Risk assessment
            col1, col2 = st.columns([2, 1])

            with col1:
                if risk_level == "High":
                    st.error(
                        f"⚠️ **HIGH RISK**: {prob:.1%} probability of attrition")
                    st.markdown("### 🚨 Recommended Actions:")
                    st.markdown(
                        "- 🗓️ Schedule **immediate** retention conversation")
                    st.markdown("- 💰 Review compensation and benefits package")
                    st.markdown("- 📈 Discuss career advancement opportunities")
                    st.markdown("- 🎯 Consider promotion or role adjustment")
                    st.markdown("- 🤝 Assign mentor or career coach")
                elif risk_level == "Medium":
                    st.warning(
                        f"⚡ **MEDIUM RISK**: {prob:.1%} probability of attrition")
                    st.markdown("### ⚠️ Recommended Actions:")
                    st.markdown("- 📊 Monitor engagement metrics closely")
                    st.markdown("- 💬 Schedule quarterly check-ins")
                    st.markdown(
                        "- 🎓 Offer professional development opportunities")
                    st.markdown(
                        "- 🔍 Address any satisfaction concerns proactively")
                else:
                    st.success(
                        f"✅ **LOW RISK**: {prob:.1%} probability of attrition")
                    st.markdown("### 👍 Status:")
                    st.markdown(
                        "Employee appears satisfied and engaged. Continue regular check-ins and recognition.")

            with col2:
                st.metric("Risk Level", risk_level, delta=None)
                st.metric("Probability", f"{prob:.1%}", delta=None)

            # Top risk factors
            if result.get('top_risk_factors'):
                st.markdown("---")
                st.subheader("🔍 Top Risk Factors")
                for i, factor in enumerate(result['top_risk_factors'][:5], 1):
                    # Clean up factor name
                    clean_factor = factor.replace('_', ' ').title()
                    st.markdown(f"**{i}.** {clean_factor}")


def show_feature_importance():
    """Display feature importance analysis."""
    st.header("📈 Feature Importance Analysis")
    st.markdown("Understanding which factors most influence attrition")

    try:
        # Load feature importance
        import pandas as pd
        from pathlib import Path

        importance_path = Path('docs/feature_importances.csv')

        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)

            st.subheader("Top 20 Most Important Features")

            # Create bar chart
            top_20 = importance_df.head(20)
            fig = px.bar(
                top_20,
                x='importance',
                y='feature',
                orientation='h',
                labels={'importance': 'Importance Score',
                        'feature': 'Feature'},
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=600,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show table
            st.subheader("📊 Complete Feature Rankings")
            st.dataframe(
                importance_df.style.background_gradient(
                    subset=['importance'], cmap='YlOrRd'),
                use_container_width=True,
                height=400
            )

        else:
            st.warning(
                "Feature importance data not found. Run: `python src/models/analyze_features.py`")

    except Exception as e:
        st.error(f"Error loading feature importance: {e}")


if __name__ == "__main__":
    main()
