"""Improved Sepsis model with better imbalance handling (SMOTE/ADASYN comparison)."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import joblib


def train_sepsis_model(data_path: Path, model_path: Path, scaler_path: Path):
    """Train sepsis model with the best oversampling method."""
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']

    print(f"Dataset: {len(df):,} patients")
    print(f"Mortalité: {y.sum():,} ({y.mean() * 100:.2f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Test different oversampling methods
    samplers = {
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
    }

    best_auc = 0
    best_model = None
    best_scaler = None
    best_method = None

    for method_name, sampler in samplers.items():
        print(f"\n📊 Test avec {method_name}...")

        X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
        print(f"   Après sampling: {len(X_train_res)} samples")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42,
        )
        model.fit(X_train_scaled, y_train_res)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"   AUC-ROC: {auc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_scaler = scaler
            best_method = method_name

    print(f"\n🏆 Meilleure méthode: {best_method} (AUC: {best_auc:.4f})")

    joblib.dump(best_model, model_path)
    joblib.dump(best_scaler, scaler_path)
    print(f"✅ Modèle sauvegardé: {model_path}")


def main():
    print("=" * 60)
    print("MODÈLE SEPSIS - VERSION AMÉLIORÉE")
    print("=" * 60)

    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    train_sepsis_model(
        data_dir / "sepsis_ready.csv",
        models_dir / "sepsis_model_improved.pkl",
        models_dir / "sepsis_scaler_improved.pkl",
    )


if __name__ == "__main__":
    main()