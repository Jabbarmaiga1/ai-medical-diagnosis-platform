"""Import du dataset hospitalier MIMIC-IV Demo depuis un fichier ZIP."""

import sys
from pathlib import Path
import zipfile
import pandas as pd


def find_csv_in_zip(zip_path: Path) -> list:
    """Retourne la liste des fichiers CSV ou CSV.gz dans le ZIP."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        return [f for f in zf.namelist() if f.endswith(('.csv', '.csv.gz'))]


def load_mimic_patients(zip_path: Path) -> pd.DataFrame:
    """
    Charge le fichier patients depuis le ZIP MIMIC-IV Demo.
    Recherche automatiquement le bon fichier.
    """
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Fichier {zip_path} non trouvé.\n"
            "Télécharge-le depuis: https://physionet.org/content/mimic-iv-demo/2.2/"
        )

    csv_files = find_csv_in_zip(zip_path)
    patient_files = [f for f in csv_files if 'patient' in f.lower()]

    if not patient_files:
        raise ValueError(
            f"Aucun fichier patient trouvé dans le ZIP.\n"
            f"Fichiers disponibles: {csv_files[:5]}"
        )

    target_file = patient_files[0]
    print(f"📁 Fichier trouvé: {target_file}")

    # Lecture avec gestion du .gz
    if target_file.endswith('.gz'):
        df = pd.read_csv(zip_path, compression='zip', file_name=target_file)
    else:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(target_file) as f:
                df = pd.read_csv(f)

    return df


def main():
    print("=" * 60)
    print("CONNEXION AU DATASET HOSPITALIER OFFICIEL")
    print("=" * 60)

    zip_path = Path("data/mimic-iv-demo-2.2.zip")

    try:
        df_patients = load_mimic_patients(zip_path)
        print(f"\n✅ Succès ! {len(df_patients)} patients chargés.")
        print("\nAperçu des données:")
        print(df_patients.head())

        print("\n✨ Données cliniques réelles chargées avec succès.")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()