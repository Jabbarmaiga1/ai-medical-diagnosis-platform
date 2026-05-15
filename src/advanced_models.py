"""Medical ML models training with hyperparameter optimization and evaluation."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


class MedicalModeling:
    """Train and evaluate multiple ML models for medical diagnosis."""

    def __init__(self):
        self.models_dir = Path("models")
        self.reports_dir = Path("reports/figures")
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.reports_dir.mkdir(exist_ok=True, parents=True)

    def load_data(self, filepath: str):
        """Load dataset and separate features/target."""
        df = pd.read_csv(filepath)
        return df.drop(columns=['target']), df['target']

    def get_models(self) -> dict:
        """Return base models configuration."""
        return {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
            'LightGBM': LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
            'SVM': SVC(probability=True, random_state=42)
        }

    def _get_param_grid(self, model_name: str) -> dict:
        """Return hyperparameter grid for supported models."""
        grids = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'num_leaves': [31, 50],
                'learning_rate': [0.01, 0.1]
            }
        }
        return grids.get(model_name, {})

    def _tune_model(self, model_name: str, model, X_train, y_train):
        """Run GridSearchCV for supported models."""
        param_grid = self._get_param_grid(model_name)
        if not param_grid:
            return model

        print(f"   Tuning {model_name}...")
        grid = GridSearchCV(
            model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        grid.fit(X_train, y_train)
        print(f"   Best params: {grid.best_params_}")
        print(f"   Best CV score: {grid.best_score_:.4f}")
        return grid.best_estimator_

    def _plot_confusion_matrix(self, y_test, y_pred, disease_name: str):
        """Save confusion matrix plot."""
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Sain/Bénin', 'Malade/Malin'],
                    yticklabels=['Sain/Bénin', 'Malade/Malin'])
        ax.set_xlabel('Prédiction')
        ax.set_ylabel('Réel')
        ax.set_title(f'Matrice de Confusion - {disease_name}')
        plt.tight_layout()
        plt.savefig(self.reports_dir / f'confusion_matrix_{disease_name}.png', dpi=300)
        plt.close()

    def train_all_diseases(self) -> dict:
        """Train models for all three diseases."""
        diseases = {
            'diabetes': {'file': 'data/diabetes.csv', 'name': 'Diabète'},
            'breast_cancer': {'file': 'data/breast_cancer.csv', 'name': 'Cancer du Sein'},
            'heart_disease': {'file': 'data/heart_disease.csv', 'name': 'Maladie Cardiaque'}
        }

        all_results = {}

        for disease_key, info in diseases.items():
            print("\n" + "=" * 70)
            print(f"TRAITEMENT: {info['name'].upper()}")
            print("=" * 70)

            X, y = self.load_data(info['file'])
            print(f"Dataset: {X.shape[0]} patients, {X.shape[1]} features")
            print(f"   Classe 0: {(y == 0).sum()}, Classe 1: {(y == 1).sum()}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = self.get_models()
            results = []

            print("\nÉVALUATION DES MODÈLES:")
            print("-" * 70)

            for name, base_model in models.items():
                model = self._tune_model(name, base_model, X_train_scaled, y_train)
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]

                metrics = {
                    'Model': name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1-Score': f1_score(y_test, y_pred),
                    'AUC-ROC': roc_auc_score(y_test, y_proba)
                }

                cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                            cv=StratifiedKFold(5), scoring='roc_auc')
                metrics['CV Mean'] = cv_scores.mean()
                metrics['CV Std'] = cv_scores.std()

                results.append(metrics)
                print(f"   {name:20} | AUC: {metrics['AUC-ROC']:.4f} | "
                      f"F1: {metrics['F1-Score']:.4f} | CV: {metrics['CV Mean']:.4f}")

            results_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)
            results_df.to_csv(self.models_dir / f'{disease_key}_comparison.csv', index=False)

            best_model_name = results_df.iloc[0]['Model']
            best_model = models[best_model_name]
            best_model = self._tune_model(best_model_name, best_model, X_train_scaled, y_train)
            best_model.fit(X_train_scaled, y_train)

            y_pred_final = best_model.predict(X_test_scaled)
            self._plot_confusion_matrix(y_test, y_pred_final, disease_key)

            report = classification_report(y_test, y_pred_final,
                                          target_names=['Sain/Bénin', 'Malade/Malin'])

            joblib.dump(best_model, self.models_dir / f'{disease_key}_best_model.pkl')
            joblib.dump(scaler, self.models_dir / f'{disease_key}_scaler.pkl')
            joblib.dump(X.columns.tolist(), self.models_dir / f'{disease_key}_features.pkl')

            all_results[disease_key] = {
                'best_model': best_model_name,
                'best_score': results_df.iloc[0]['AUC-ROC'],
                'results': results_df,
                'classification_report': report
            }

            # Save markdown report
            report_path = Path('reports') / f'{disease_key}_model_report.md'
            report_path.parent.mkdir(exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# {info['name']} - Rapport de Modélisation\n\n")
                f.write(f"**Meilleur Modèle:** {best_model_name}\n\n")
                f.write(f"**AUC-ROC:** {results_df.iloc[0]['AUC-ROC']:.4f}\n\n")
                f.write("### Comparaison des Modèles\n")
                f.write(results_df.to_markdown(index=False))
                f.write("\n\n### Classification Report\n")
                f.write(f"```\n{report}\n```\n")

            print(f"\n✅ Modèle sauvegardé: {disease_key}_best_model.pkl")

        # Summary
        summary = pd.DataFrame([{
            'Disease': d,
            'Best Model': info['best_model'],
            'AUC-ROC': info['best_score']
        } for d, info in all_results.items()])
        summary.to_csv(self.models_dir / 'summary_results.csv', index=False)

        print("\n" + "=" * 70)
        print("MODÉLISATION TERMINÉE")
        print("=" * 70)
        print("\nRÉSUMÉ FINAL:")
        print(summary.to_string(index=False))

        return all_results


if __name__ == "__main__":
    modeling = MedicalModeling()
    modeling.train_all_diseases()