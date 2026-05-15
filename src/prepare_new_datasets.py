"""Préparation des datasets Cancer Pancréas, Sepsis et Alzheimer pour l'entraînement."""

import sys
from pathlib import Path
import pandas as pd


def safe_read_csv(filepath: Path) -> pd.DataFrame:
    """Read CSV with error handling."""
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    return pd.read_csv(filepath)


def prepare_pancreatic_cancer(data_dir: Path):
    """Prepare pancreatic cancer dataset with binary target."""
    input_path = data_dir / "Pancreatic_Cancer.csv"
    output_path = data_dir / "pancreatic_cancer_ready.csv"

    df = safe_read_csv(input_path)
    print(f"Avant: {df.shape}")

    df['target'] = (df['Deaths'] > df['Deaths'].median()).astype(int)

    feature_cols = ['Cancer_Sites_Code', 'Age_Group_Code', 'Sex_Code', 'Race_Code']
    df_clean = df[feature_cols + ['target']].copy()

    for col in feature_cols:
        df_clean[col] = df_clean[col].astype('category').cat.codes

    df_clean.to_csv(output_path, index=False)
    print(f"Après: {df_clean.shape}")
    print(f"Target: {df_clean['target'].value_counts().to_dict()}")
    print(f"✅ Sauvegardé: {output_path}")


def prepare_sepsis(data_dir: Path):
    """Prepare sepsis dataset with mortality target."""
    possible_names = [
        "sepsis_minimal_EHRs_Norway.csv",
        "sepsis_norway.csv",
        "sepsis_uci.csv"
    ]

    input_path = None
    for name in possible_names:
        candidate = data_dir / name
        if candidate.exists():
            input_path = candidate
            break

    if not input_path:
        raise FileNotFoundError(f"Aucun fichier sepsis trouvé parmi {possible_names}")

    output_path = data_dir / "sepsis_ready.csv"

    df = safe_read_csv(input_path)
    print(f"Avant: {df.shape}")

    # Renommage sécurisé
    if len(df.columns) >= 5:
        df.columns = ['age', 'sex', 'length_of_stay', 'target', 'septic_episode']
    else:
        raise ValueError(f"Format inattendu: {df.columns.tolist()}")

    df_clean = df[['age', 'sex', 'length_of_stay', 'target']].dropna()
    df_clean['target'] = df_clean['target'].astype(int)

    df_clean.to_csv(output_path, index=False)
    print(f"Après: {df_clean.shape}")
    print(f"Target (mortalité): {df_clean['target'].sum()} ({df_clean['target'].mean()*100:.2f}%)")
    print(f"✅ Sauvegardé: {output_path}")


def prepare_alzheimer(data_dir: Path):
    """Prepare Alzheimer DARWIN dataset."""
    possible_names = ["data.csv", "darwin.csv"]
    input_path = None
    for name in possible_names:
        candidate = data_dir / name
        if candidate.exists():
            input_path = candidate
            break

    if not input_path:
        raise FileNotFoundError(f"Aucun fichier Alzheimer trouvé parmi {possible_names}")

    output_path = data_dir / "alzheimer_ready.csv"

    df = safe_read_csv(input_path)
    print(f"Avant: {df.shape}")

    if 'class' not in df.columns:
        raise ValueError(f"Colonne 'class' manquante. Colonnes disponibles: {df.columns.tolist()}")

    class_values = df['class'].unique()
    print(f"Classes trouvées: {class_values}")

    if 'P' in class_values and 'H' in class_values:
        df['target'] = (df['class'] == 'P').astype(int)
    else:
        df['target'] = (df['class'] == 'AD').astype(int)

    df_clean = df.drop(['ID', 'class'], axis=1)

    df_clean.to_csv(output_path, index=False)
    print(f"Après: {df_clean.shape}")
    print(f"Target: Patients: {df_clean['target'].sum()}, Healthy: {len(df_clean) - df_clean['target'].sum()}")
    print(f"✅ Sauvegardé: {output_path}")


def main():
    print("=" * 60)
    print("PRÉPARATION DES 3 DATASETS")
    print("=" * 60)

    data_dir = Path("data")

    try:
        print("\n🩺 1. CANCER PANCRÉAS")
        print("-" * 40)
        prepare_pancreatic_cancer(data_dir)

        print("\n🚑 2. SEPSIS")
        print("-" * 40)
        prepare_sepsis(data_dir)

        print("\n🧠 3. ALZHEIMER (DARWIN)")
        print("-" * 40)
        prepare_alzheimer(data_dir)

        print("\n" + "=" * 60)
        print("RÉSUMÉ DES DATASETS PRÊTS")
        print("=" * 60)

        for name, path in [
            ("Cancer Pancréas", "pancreatic_cancer_ready.csv"),
            ("Sepsis", "sepsis_ready.csv"),
            ("Alzheimer", "alzheimer_ready.csv"),
        ]:
            df = pd.read_csv(data_dir / path)
            print(f"✅ {name:20} : {len(df):5} lignes, {df.shape[1] - 1:2} features")

        print("\n" + "=" * 60)
        print("DATASETS PRÊTS POUR L'ENTRAÎNEMENT")
        print("=" * 60)

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()