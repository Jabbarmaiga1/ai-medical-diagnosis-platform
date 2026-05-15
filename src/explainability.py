"""Explainable AI - Feature importance with ELI5 permutation importance."""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import PermutationImportance
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class ExplainableMedicalAI:
    """Feature importance analysis for medical diagnosis models."""

    def __init__(self):
        self.models_dir = Path("models")
        self.data_dir = Path("data")
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)

        self.models = {}
        self.scalers = {}
        self._load_models()

    def _load_models(self):
        """Load trained models and scalers."""
        diseases = ['diabetes', 'breast_cancer', 'heart_disease']

        for disease in diseases:
            model_path = self.models_dir / f'{disease}_best_model.pkl'
            scaler_path = self.models_dir / f'{disease}_scaler.pkl'

            if model_path.exists() and scaler_path.exists():
                self.models[disease] = joblib.load(model_path)
                self.scalers[disease] = joblib.load(scaler_path)
                print(f"✅ Modèle {disease} chargé")
            else:
                print(f"⚠️ Modèle {disease} non trouvé")

    def _get_data(self, disease_name: str):
        """Load and scale data for a given disease."""
        df = pd.read_csv(self.data_dir / f'{disease_name}.csv')
        X = df.drop(columns=['target'])
        y = df['target']
        X_scaled = self.scalers[disease_name].transform(X)
        return X, X_scaled, y

    def get_feature_importance(self, disease_name: str):
        """Compute permutation feature importance."""
        if disease_name not in self.models:
            print(f"❌ Modèle {disease_name} non chargé")
            return None

        X, X_scaled, y = self._get_data(disease_name)
        feature_names = X.columns.tolist()

        # Permutation importance
        perm = PermutationImportance(self.models[disease_name], random_state=42, n_jobs=-1)
        perm.fit(X_scaled, y)

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm.feature_importances_
        }).sort_values('importance', ascending=True)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance_df['feature'], importance_df['importance'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance - {disease_name}')
        plt.tight_layout()
        plt.savefig(self.reports_dir / f'feature_importance_{disease_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

        return importance_df

    def generate_feature_importance_report(self):
        """Generate feature importance reports for all diseases."""
        diseases = ['diabetes', 'breast_cancer', 'heart_disease']

        for disease in diseases:
            print(f"\n📊 Analyse pour {disease.upper()}")
            print("-" * 50)

            importance = self.get_feature_importance(disease)
            if importance is not None:
                print(importance.tail(10).to_string())
                importance.to_csv(self.reports_dir / f'feature_importance_{disease}.csv', index=False)

        print("\n✅ Rapports générés dans reports/")


if __name__ == "__main__":
    explainer = ExplainableMedicalAI()
    explainer.generate_feature_importance_report()