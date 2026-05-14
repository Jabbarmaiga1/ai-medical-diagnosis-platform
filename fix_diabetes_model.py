"""Retrain diabetes model with PIMA dataset (8 features)."""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import urllib.request
import urllib.error


PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMNS = [
    'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
    'insulin', 'bmi', 'diabetes_pedigree', 'age', 'target'
]


def download_pima_dataset(url: str, timeout: int = 30) -> pd.DataFrame:
    """Download PIMA diabetes dataset from GitHub."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            df = pd.read_csv(response, names=COLUMNS)
        return df
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        raise RuntimeError(f"Failed to download dataset from {url}: {e}")


def train_diabetes_model(data_path: Path = None, model_dir: Path = None):
    """Train and save diabetes model with PIMA dataset."""
    if model_dir is None:
        model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("RÉENTRAÎNEMENT MODÈLE DIABÈTE (8 features)")
    print("=" * 60)

    print("\n📥 Téléchargement du dataset PIMA Diabetes...")
    df = download_pima_dataset(PIMA_URL)
    print(f"✅ Dataset chargé: {len(df)} patients, {len(df.columns) - 1} features")

    print("\n📊 Aperçu des données:")
    print(df.head())

    X = df.drop('target', axis=1)
    y = df['target']

    print(f"\n📈 Distribution:")
    print(f"   - Non-diabétiques: {(y == 0).sum()}")
    print(f"   - Diabétiques: {(y == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n🤖 Entraînement du modèle...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n📊 Performance:")
    print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - AUC-ROC: {auc:.4f}")

    joblib.dump(model, model_dir / "diabetes_best_model.pkl")
    joblib.dump(scaler, model_dir / "diabetes_scaler.pkl")
    joblib.dump(X.columns.tolist(), model_dir / "diabetes_features.pkl")

    print("\n✅ Modèle sauvegardé!")
    print(f"📁 {model_dir / 'diabetes_best_model.pkl'}")
    print(f"📁 {model_dir / 'diabetes_scaler.pkl'}")
    print(f"📁 {model_dir / 'diabetes_features.pkl'}")

    print("\n" + "=" * 60)
    print("MODÈLE DIABÈTE PRÊT (8 features)")
    print("=" * 60)
    print("\n📋 Features utilisées:")
    for i, feat in enumerate(X.columns):
        print(f"   {i + 1}. {feat}")


def main():
    try:
        train_diabetes_model()
    except RuntimeError as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()