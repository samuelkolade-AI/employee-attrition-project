"""
Prediction service for attrition risk scoring
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List
import json


class AttritionPredictor:
    """Load model and make predictions."""

    def __init__(self):
        self.model_dir = Path('models/production')
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_artifacts()

    def load_artifacts(self):
        """Load model, scaler, and feature names."""
        # Find the best model
        model_files = list(self.model_dir.glob('*_model.pkl'))
        if not model_files:
            raise FileNotFoundError("No model found in models/production/")

        model_path = model_files[0]
        self.model = joblib.load(model_path)
        print(f"✅ Loaded model: {model_path.name}")

        # Load scaler
        scaler_path = self.model_dir / 'scaler.pkl'
        self.scaler = joblib.load(scaler_path)
        print(f"✅ Loaded scaler")

        # Load feature names
        feature_path = self.model_dir / 'feature_names.txt'
        with open(feature_path, 'r') as f:
            self.feature_names = [line.strip() for line in f]
        print(f"✅ Loaded {len(self.feature_names)} feature names")

    def predict_single(self, features: Dict) -> Dict:
        """
        Predict attrition risk for a single employee.

        Args:
            features: Dictionary of employee features

        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Ensure all required features are present
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0  # Default missing features to 0

        # Select and order features
        X = df[self.feature_names]

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        probability = self.model.predict_proba(X_scaled)[0, 1]
        prediction = int(probability > 0.5)

        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Get top risk factors
        top_factors = self._get_top_risk_factors(X_scaled[0])

        return {
            'attrition_probability': float(probability),
            'prediction': prediction,
            'risk_level': risk_level,
            'top_risk_factors': top_factors
        }

    def _get_top_risk_factors(self, feature_values, top_n=3) -> List[str]:
        """Identify top risk factors based on feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            contributions = self.model.feature_importances_ * \
                np.abs(feature_values)
            top_indices = np.argsort(contributions)[-top_n:][::-1]
            return [self.feature_names[i] for i in top_indices]
        return []


# Test
if __name__ == "__main__":
    predictor = AttritionPredictor()

    test_features = {
        'age': 35,
        'tenure_years': 5,
        'monthly_income': 5000,
        'satisfaction_score': 2.5,
        'overtime_flag': 1,
    }

    result = predictor.predict_single(test_features)
    print("\n📊 Prediction Result:")
    print(json.dumps(result, indent=2))
