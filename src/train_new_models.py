"""Train ML models for Pancreatic Cancer, Sepsis and Alzheimer."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib


def train_model(
    data_path: Path,
    model_path: Path,
    scaler_path: Path,
    model_name: str,
    use_smote: bool = False,
    model_type: str = "rf",
):
    """Generic training function for all diseases."""
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']

    print(f"Dataset: {len(df):,} patients, {X.shape[1]} features")
    if use_smote:
        print(f"Distribution: 0={(y == 0).sum():,}, 1={(y == 1).sum():,} ({y.mean() * 100:.2f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_proba):.4f}")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Modèle sauvegardé: {model_path}")


def main():
    print("=" * 60)
    print("ENTRAÎNEMENT DES MODÈLES - NOUVELLES MALADIES")
    print("=" * 60)

    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    try:
        print("\n🩺 1. CANCER PANCRÉAS")
        print("-" * 40)
        train_model(
            data_dir / "pancreatic_cancer_ready.csv",
            models_dir / "pancreatic_cancer_model.pkl",
            models_dir / "pancreatic_cancer_scaler.pkl",
            model_name="pancreatic_cancer",
            use_smote=False,
            model_type="rf",
        )

        print("\n🚑 2. SEPSIS")
        print("-" * 40)
        train_model(
            data_dir / "sepsis_ready.csv",
            models_dir / "sepsis_model.pkl",
            models_dir / "sepsis_scaler.pkl",
            model_name="sepsis",
            use_smote=True,
            model_type="xgb",
        )

        print("\n🧠 3. ALZHEIMER")
        print("-" * 40)
        train_model(
            data_dir / "alzheimer_ready.csv",
            models_dir / "alzheimer_model.pkl",
            models_dir / "alzheimer_scaler.pkl",
            model_name="alzheimer",
            use_smote=False,
            model_type="xgb",
        )

        print("\n" + "=" * 60)
        print("🎉 3 NOUVEAUX MODÈLES ENTRAÎNÉS AVEC SUCCÈS")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()