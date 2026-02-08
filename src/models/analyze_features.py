"""
Analyze feature importance using SHAP
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def analyze_feature_importance():
    """Generate feature importance plots."""

    print("🔍 Analyzing feature importance...")

    try:
        # Load best model (find it automatically)
        model_dir = Path('models/production')
        model_files = list(model_dir.glob('*_model.pkl'))

        if not model_files:
            print("❌ No trained model found. Please run train_model.py first.")
            return False

        model_path = model_files[0]
        model = joblib.load(model_path)
        print(f"✅ Loaded model: {model_path.name}")

        # Load feature names
        feature_path = model_dir / 'feature_names.txt'
        with open(feature_path, 'r') as f:
            feature_names = [line.strip() for line in f]

        # Feature Importance from model
        if hasattr(model, 'feature_importances_'):
            print("\n📊 Top 20 Most Important Features:")
            importances = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(importances.head(20).to_string(index=False))

            # Save to CSV
            docs_dir = Path('docs')
            docs_dir.mkdir(exist_ok=True)

            importances.to_csv(
                docs_dir / 'feature_importances.csv', index=False)
            print(f"\n✅ Saved: docs/feature_importances.csv")

            # Plot
            plt.figure(figsize=(10, 8))
            top_20 = importances.head(20)
            plt.barh(range(len(top_20)), top_20['importance'])
            plt.yticks(range(len(top_20)), top_20['feature'])
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(docs_dir / 'feature_importance_plot.png',
                        dpi=300, bbox_inches='tight')
            print(f"✅ Saved: docs/feature_importance_plot.png")
            plt.close()

        else:
            print("⚠️  This model doesn't support feature_importances_")

        print("\n✅ Feature analysis complete!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    analyze_feature_importance()
