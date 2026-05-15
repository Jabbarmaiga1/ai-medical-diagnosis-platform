"""Téléchargement des datasets médicaux depuis UCI et scikit-learn."""

import pandas as pd
import urllib.request
import urllib.error
from pathlib import Path
from sklearn.datasets import load_breast_cancer, load_diabetes


def download_heart_disease(data_dir: Path) -> pd.DataFrame:
    """Télécharge et sauvegarde le dataset Heart Disease (Cleveland)."""
    print("📊 Téléchargement Heart Disease Dataset...")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    try:
        df = pd.read_csv(url, names=column_names, na_values='?', timeout=30)
    except (urllib.error.URLError, pd.errors.ParserError) as e:
        raise RuntimeError(f"Erreur lors du téléchargement du dataset heart disease: {e}")

    df = df.dropna()
    df['target'] = (df['target'] > 0).astype(int)

    output_path = data_dir / "heart_disease.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Heart Disease: {len(df)} patients, {len(df.columns) - 1} features")
    print(f"   - Malades: {df['target'].sum()}")
    print(f"   - Sain: {len(df) - df['target'].sum()}")

    return df


def download_breast_cancer(data_dir: Path) -> pd.DataFrame:
    """Charge et sauvegarde le dataset Breast Cancer Wisconsin."""
    print("\n📊 Téléchargement Breast Cancer Dataset...")

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    output_path = data_dir / "breast_cancer.csv"
    df.to_csv(output_path, index=False)

    benign = (df['target'] == 1).sum()
    malignant = (df['target'] == 0).sum()
    print(f"✅ Breast Cancer: {len(df)} patients, {len(df.columns) - 1} features")
    print(f"   - Bénin: {benign}")
    print(f"   - Malin: {malignant}")

    return df


def download_diabetes(data_dir: Path) -> pd.DataFrame:
    """Charge et sauvegarde le dataset Diabetes (scikit-learn)."""
    print("\n📊 Téléchargement Diabetes Dataset...")

    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = (data.target > 140).astype(int)

    output_path = data_dir / "diabetes.csv"
    df.to_csv(output_path, index=False)

    diabetic = df['target'].sum()
    non_diabetic = len(df) - diabetic
    print(f"✅ Diabetes: {len(df)} patients, {len(df.columns) - 1} features")
    print(f"   - Diabétiques: {diabetic}")
    print(f"   - Non-diabétiques: {non_diabetic}")

    return df


def main():
    print("=" * 60)
    print("TÉLÉCHARGEMENT DATASETS MÉDICAUX")
    print("=" * 60)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    heart = download_heart_disease(data_dir)
    cancer = download_breast_cancer(data_dir)
    diabetes = download_diabetes(data_dir)

    print("\n" + "=" * 60)
    print("TOUS LES DATASETS SONT PRÊTS")
    print("=" * 60)

    print("\nAperçu Heart Disease:")
    print(heart.head())


if __name__ == "__main__":
    main()