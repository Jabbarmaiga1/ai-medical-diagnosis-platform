"""Analyse des datasets Cancer Pancréas, Alzheimer et Sepsis."""

import pandas as pd
import glob
from pathlib import Path


def find_csv_file(directory: str, keywords: list) -> Path | None:
    """Recherche un fichier CSV contenant l'un des mots-clés."""
    for filepath in glob.glob(f"{directory}/**/*.csv", recursive=True):
        if any(keyword in Path(filepath).stem.lower() for keyword in keywords):
            return Path(filepath)
    return None


def analyze_dataset(filepath: Path, name: str, target_col: str = None):
    """Affiche les informations et statistiques d'un dataset."""
    print(f"\n{name}")
    print("-" * 40)

    if not filepath or not filepath.exists():
        print(f"❌ Fichier non trouvé")
        return None

    try:
        df = pd.read_csv(filepath)
        print(f"✅ {len(df):,} lignes, {len(df.columns)} colonnes")
        print(f"📁 {filepath}")

        cols = df.columns.tolist()
        print(f"\n📋 Colonnes ({len(cols)}):")
        print(", ".join(cols[:8]) + ("..." if len(cols) > 8 else ""))

        print(f"\n📊 Aperçu:")
        print(df.head(3).to_string())

        if target_col and target_col in df.columns:
            print(f"\n📈 Distribution de '{target_col}':")
            print(df[target_col].value_counts().to_dict())

        return df

    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None


def main():
    print("=" * 60)
    print("ANALYSE DES 3 NOUVEAUX DATASETS")
    print("=" * 60)

    data_dir = "data"

    # Cancer Pancréas
    pancreatic_file = find_csv_file(data_dir, ['pancan', 'pancreatic'])
    analyze_dataset(pancreatic_file, "🩺 CANCER PANCRÉAS", target_col=None)

    # Sepsis
    sepsis_file = Path("data/sepsis_minimal_EHRs_Norway.csv")
    if not sepsis_file.exists():
        sepsis_file = find_csv_file(data_dir, ['sepsis', 'norway'])
    analyze_dataset(sepsis_file, "🚑 SEPSIS (Norway EHRs)", target_col='in-hospital_death')

    # Alzheimer
    alzheimer_file = find_csv_file(data_dir, ['darwin', 'data'])
    analyze_dataset(alzheimer_file, "🧠 ALZHEIMER (DARWIN)", target_col=None)

    print("\n" + "=" * 60)
    print("Analyse terminée")
    print("=" * 60)


if __name__ == "__main__":
    main()