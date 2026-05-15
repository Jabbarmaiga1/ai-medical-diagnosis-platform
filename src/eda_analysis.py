"""Analyse exploratoire des données pour les 3 datasets médicaux."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(data_dir: Path):
    """Charge les trois datasets depuis le dossier data."""
    return (
        pd.read_csv(data_dir / "heart_disease.csv"),
        pd.read_csv(data_dir / "breast_cancer.csv"),
        pd.read_csv(data_dir / "diabetes.csv"),
    )


def save_descriptive_stats(heart, cancer, diabetes, output_dir: Path):
    """Génère un rapport markdown des statistiques descriptives."""
    with open(output_dir / "data_description.md", "w", encoding="utf-8") as f:
        f.write("# 📊 Dataset Description - AI Medical Diagnosis\n\n")
        f.write("## Overview\n")
        f.write("This report provides a comprehensive analysis of three medical datasets.\n\n")

        for name, df in zip(
            ["Heart Disease", "Breast Cancer", "Diabetes"],
            [heart, cancer, diabetes],
        ):
            f.write(f"\n## {name}\n")
            f.write(f"- **Number of patients:** {len(df)}\n")
            f.write(f"- **Number of features:** {len(df.columns) - 1}\n")
            f.write(f"- **Target distribution:**\n")
            f.write(
                f"  - Class 0: {(df['target'] == 0).sum()} ({(df['target'] == 0).mean() * 100:.1f}%)\n"
            )
            f.write(
                f"  - Class 1: {(df['target'] == 1).sum()} ({(df['target'] == 1).mean() * 100:.1f}%)\n"
            )
            f.write("\n### Feature Statistics\n")
            f.write(df.describe().to_html())
            f.write("\n\n---\n")


def plot_disease_distribution(heart, cancer, diabetes, output_dir: Path):
    """Barplot de la distribution des trois pathologies."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    diseases_data = [
        (heart, "Heart Disease", ["Sain", "Malade"]),
        (cancer, "Breast Cancer", ["Bénin", "Malin"]),
        (diabetes, "Diabetes", ["Non-diabétique", "Diabétique"]),
    ]

    for i, (df, title, labels) in enumerate(diseases_data):
        counts = df["target"].value_counts().sort_index()
        axes[i].bar(labels, counts.values, color=["green", "red"], alpha=0.7)
        axes[i].set_title(f"{title} Distribution", fontsize=12, fontweight="bold")
        axes[i].set_ylabel("Number of Patients")

        for j, v in enumerate(counts.values):
            axes[i].text(j, v + 5, str(v), ha="center", fontweight="bold")

    plt.suptitle("Distribution des Maladies", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "disease_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation_analysis(heart, cancer, diabetes, output_dir: Path):
    """Affiche les corrélations des features avec la target."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (df, title) in enumerate(
        zip([heart, cancer, diabetes], ["Heart Disease", "Breast Cancer", "Diabetes"])
    ):
        corr_with_target = df.corr(numeric_only=True)["target"].drop("target").sort_values(ascending=False)

        colors = ["red" if x < 0 else "green" for x in corr_with_target.values]
        axes[i].barh(range(len(corr_with_target)), corr_with_target.values, color=colors, alpha=0.7)
        axes[i].set_yticks(range(len(corr_with_target)))
        axes[i].set_yticklabels(corr_with_target.index, fontsize=8)
        axes[i].set_xlabel("Corrélation avec Target")
        axes[i].set_title(f"{title}\nFeature Importance (Corrélation)", fontsize=10, fontweight="bold")
        axes[i].axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "correlation_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_features_boxplot(heart, output_dir: Path):
    """Boxplots des 3 features les plus corrélées (Heart Disease)."""
    top_features = heart.corr(numeric_only=True)["target"].drop("target").abs().nlargest(3).index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, feature in enumerate(top_features):
        heart.boxplot(column=feature, by="target", ax=axes[i])
        axes[i].set_title(f"{feature} by Target", fontsize=10)
        axes[i].set_xlabel("Target (0=Sain, 1=Malade)")

    plt.suptitle("Heart Disease - Top 3 Features Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "top_features_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_html_report(heart, cancer, diabetes, output_dir: Path):
    """Crée un rapport HTML complet avec toutes les sections."""
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>AI Medical Diagnosis - EDA Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; }}
        img {{ max-width: 100%; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 20px; padding: 20px; background: #ecf0f1; border-radius: 10px; }}
    </style>
</head>
<body>
    <h1>🏥 AI Medical Diagnosis - Exploratory Data Analysis</h1>
    <div>
        <div class="metric">📊 Heart Disease: {len(heart)} patients</div>
        <div class="metric">🎗️ Breast Cancer: {len(cancer)} patients</div>
        <div class="metric">🩺 Diabetes: {len(diabetes)} patients</div>
    </div>
    <h2>1. Disease Distribution</h2>
    <img src="figures/disease_distribution.png">
    <h2>2. Feature Correlation Analysis</h2>
    <img src="figures/correlation_analysis.png">
    <h2>3. Top Features Distribution</h2>
    <img src="figures/top_features_boxplot.png">
    <h2>4. Dataset Statistics</h2>
    <h3>Heart Disease</h3>
    {heart.describe().to_html()}
    <h3>Breast Cancer</h3>
    {cancer.describe().to_html()}
    <h3>Diabetes</h3>
    {diabetes.describe().to_html()}
</body>
</html>"""

    with open(output_dir / "eda_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)


def create_eda_report():
    """Génère l'ensemble du rapport EDA (figures + markdown + html)."""
    print("=" * 60)
    print("GÉNÉRATION DU RAPPORT EDA")
    print("=" * 60)

    data_dir = Path("data")
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    heart, cancer, diabetes = load_data(data_dir)

    print("\n1️⃣ Statistiques descriptives...")
    save_descriptive_stats(heart, cancer, diabetes, output_dir.parent)

    print("\n2️⃣ Distribution des maladies...")
    plot_disease_distribution(heart, cancer, diabetes, output_dir)

    print("\n3️⃣ Matrices de corrélation...")
    plot_correlation_analysis(heart, cancer, diabetes, output_dir)

    print("\n4️⃣ Boxplots des features importantes...")
    plot_top_features_boxplot(heart, output_dir)

    print("\n5️⃣ Génération du rapport HTML...")
    generate_html_report(heart, cancer, diabetes, output_dir.parent)

    print("\n" + "=" * 60)
    print("RAPPORT EDA COMPLET")
    print("=" * 60)
    print("\n📁 Fichiers générés:")
    print("   - reports/data_description.md")
    print("   - reports/figures/disease_distribution.png")
    print("   - reports/figures/correlation_analysis.png")
    print("   - reports/figures/top_features_boxplot.png")
    print("   - reports/eda_report.html")


if __name__ == "__main__":
    create_eda_report()