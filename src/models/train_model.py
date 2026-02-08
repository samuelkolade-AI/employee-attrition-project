"""
Train Employee Attrition Prediction Models
Uses multiple algorithms and tracks experiments with MLflow
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, recall_score
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Update with your actual password
DATABASE_URL = "postgresql://attrition_user:attrition_DBpass001@localhost:5432/employee_attrition"


class AttritionModelTrainer:
    """Train and evaluate attrition prediction models."""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()

    def load_data(self):
        """Load features from PostgreSQL."""
        print("📂 Loading data from PostgreSQL...")
        engine = create_engine(DATABASE_URL)
        df = pd.read_sql("SELECT * FROM ml_features", engine)

        # Separate features and target
        X = df.drop(['employee_id', 'attrition_label'], axis=1)
        y = df['attrition_label']

        print(f"✅ Loaded {len(X)} samples with {len(X.columns)} features")
        print(f"   Attrition rate: {y.mean():.2%}")

        return X, y, df['employee_id']

    def prepare_data(self, X, y, test_size=0.2, use_smote=True):
        """Split and optionally balance the dataset."""
        print("\n🔀 Splitting data...")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"   Train set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Handle class imbalance with SMOTE
        if use_smote:
            print("\n⚖️  Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(
                X_train_scaled, y_train)
            print(f"   After SMOTE: {len(X_train_scaled)} samples")
            print(f"   New attrition rate: {y_train.mean():.2%}")

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns

    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model."""
        print("\n📊 Training Logistic Regression...")

        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)

        self.models['logistic_regression'] = model
        return model

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model."""
        print("\n🌲 Training Random Forest...")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)

        self.models['random_forest'] = model
        return model

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model."""
        print("\n🚀 Training XGBoost...")

        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        self.models['xgboost'] = model
        return model

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance."""
        print(f"\n📈 Evaluating {model_name}...")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }

        # PR-AUC (important for imbalanced data)
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_test, y_pred_proba)
        metrics['pr_auc'] = auc(recall_curve, precision_curve)

        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        # Print metrics
        print(f"   Accuracy:  {metrics['accuracy']:.3f}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall:    {metrics['recall']:.3f}")
        print(f"   F1-Score:  {metrics['f1_score']:.3f}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.3f}")
        print(f"   PR-AUC:    {metrics['pr_auc']:.3f}")

        # Confusion Matrix
        cm = self.results[model_name]['confusion_matrix']
        print(f"\n   Confusion Matrix:")
        print(f"   TN: {cm[0, 0]:4d}  FP: {cm[0, 1]:4d}")
        print(f"   FN: {cm[1, 0]:4d}  TP: {cm[1, 1]:4d}")

        return metrics

    def select_best_model(self):
        """Select best model based on F1-score."""
        print("\n🏆 Selecting best model...")

        best_f1 = 0
        best_name = None

        for name, result in self.results.items():
            f1 = result['metrics']['f1_score']
            print(f"   {name}: F1={f1:.3f}")
            if f1 > best_f1:
                best_f1 = f1
                best_name = name

        self.best_model = self.models[best_name]

        print(f"\n   ⭐ Best model: {best_name}")
        print(f"   F1-Score: {best_f1:.3f}")

        return best_name, self.best_model

    def save_model(self, model, model_name, feature_names):
        """Save model and metadata."""
        print(f"\n💾 Saving {model_name}...")

        # Create models directory
        models_dir = Path('models/production')
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = models_dir / f'{model_name}_model.pkl'
        joblib.dump(model, model_path)
        print(f"   Model saved to: {model_path}")

        # Save scaler
        scaler_path = models_dir / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"   Scaler saved to: {scaler_path}")

        # Save feature names
        feature_path = models_dir / 'feature_names.txt'
        with open(feature_path, 'w') as f:
            f.write('\n'.join(feature_names))
        print(f"   Features saved to: {feature_path}")

        # Save metadata
        metadata = {
            'model_name': model_name,
            'trained_date': pd.Timestamp.now().isoformat(),
            'metrics': {k: float(v) for k, v in self.results[model_name]['metrics'].items()},
            'feature_count': len(feature_names)
        }

        import json
        metadata_path = models_dir / f'{model_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   Metadata saved to: {metadata_path}")


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("EMPLOYEE ATTRITION MODEL TRAINING")
    print("=" * 80)

    # Initialize trainer
    trainer = AttritionModelTrainer()

    # Load data
    X, y, employee_ids = trainer.load_data()

    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(
        X, y)

    # Train models
    lr_model = trainer.train_logistic_regression(X_train, y_train)
    rf_model = trainer.train_random_forest(X_train, y_train)
    xgb_model = trainer.train_xgboost(X_train, y_train)

    # Evaluate models
    lr_metrics = trainer.evaluate_model(
        lr_model, X_test, y_test, 'logistic_regression')
    rf_metrics = trainer.evaluate_model(
        rf_model, X_test, y_test, 'random_forest')
    xgb_metrics = trainer.evaluate_model(xgb_model, X_test, y_test, 'xgboost')

    # Select best model
    best_name, best_model = trainer.select_best_model()

    # Save best model
    trainer.save_model(best_model, best_name, feature_names)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Best model saved: {best_name}")
    print(f"📁 Location: models/production/")
    print(
        f"🎯 F1-Score: {trainer.results[best_name]['metrics']['f1_score']:.3f}")

    return trainer


if __name__ == "__main__":
    trainer = main()
