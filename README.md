# 🏥 AI Medical Diagnosis Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange.svg)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> ⚠️ **AVERTISSEMENT** : Projet de démonstration éducatif. Ne pas utiliser pour des diagnostics médicaux réels.

## 🎯 En bref
 
Plateforme IA qui détecte **6 pathologies** à partir de données cliniques et radiologiques.

| Pathologie | AUC-ROC | Accuracy |
|------------|---------|----------|
| 🎗️ Cancer du Sein | **0.995** | 97% |
| ❤️ Maladie Cardiaque | **0.954** | 90% |
| 🫁 Pneumonie (radio) | 0.94 | **91.5%** |
| 🫀 Cancer Pancréas | 0.852 | 78% |
| 🩸 Diabète | 0.826 | 78% |
| 🦠 Sepsis | 0.735 | 63% |

## 🚀 Lancer l'app

```bash
git clone https://github.com/votre-username/ai-medical-diagnosis-platform.git
cd ai-medical-diagnosis-platform
pip install -r requirements.txt
streamlit run dashboard/app.py

📊 Stack
ML : Scikit-learn, XGBoost, TensorFlow

Dashboard : Streamlit

Datasets : UCI, Kaggle, MIMIC-IV (données réelles)

📁 Structure
text
├── dashboard/          # Interface Streamlit
├── src/               # Code source (modèles, data, utils)
├── tests/             # Tests unitaires
├── models/            # Modèles entraînés (.pkl, .h5)
├── data/              # Datasets
└── logs/              # Logs applicatifs
📸 Demo
Diagnostic	Résultat
Formulaire patient	🖼️
Risque élevé (rouge)	🖼️
Risque faible (vert)	🖼️
Performance graph	🖼️
⚠️ Disclaimer
Projet éducatif uniquement - Consultez toujours un médecin.

📄 Licence
MIT

---

## 📋 Version plus détaillée (si tu préfères)

```markdown
# 🏥 AI Medical Diagnosis Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange.svg)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> ⚠️ **AVERTISSEMENT MÉDICAL** : Ce projet est une **démonstration éducative**.  
> Les résultats ne doivent **PAS** être utilisés pour des diagnostics médicaux réels.  
> Consultez toujours un professionnel de santé.

---

## 🎯 Pathologies détectables

| Pathologie | Type de données | Modèle | AUC-ROC | Accuracy |
|------------|----------------|--------|---------|----------|
| 🩸 Diabète | Cliniques (8 features) | Logistic Regression | 0.826 | 78% |
| 🎗️ Cancer du Sein | Cytologie (30 features) | Logistic Regression | **0.995** | 97% |
| ❤️ Maladie Cardiaque | Cliniques (13 features) | SVM | **0.954** | 90% |
| 🫀 Cancer Pancréas | Démographique (4 features) | Random Forest | 0.852 | 78% |
| 🦠 Sepsis | Cliniques (3 features) | XGBoost + SMOTE | 0.735 | 63% |
| 🫁 Pneumonie | Radiographie (224x224) | CNN (ResNet50) | 0.94 | **91.5%** |

---

## 🏗️ Architecture
ai_medical_diagnosis_pro/
│
├── dashboard/
│ └── app.py # Interface Streamlit
│
├── src/
│ ├── config.py # Configuration et constantes
│ ├── data/
│ │ ├── loader.py # Chargement des datasets
│ │ └── preprocessor.py # Prétraitement
│ ├── models/
│ │ ├── trainer.py # Entraînement
│ │ └── predictor.py # Prédiction
│ └── utils/
│ ├── logging_config.py # Logging
│ └── security.py # Sécurité (validation inputs)
│
├── tests/ # Tests unitaires
├── models/ # Modèles entraînés (.pkl, .h5)
├── data/ # Datasets
├── logs/ # Logs applicatifs
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

text

---

## 🚀 Installation et exécution

### Prérequis
- Python 3.12+
- pip

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-username/ai-medical-diagnosis-platform.git
cd ai-medical-diagnosis-platform

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate      # Linux/Mac
# ou
venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le dashboard
streamlit run dashboard/app.py
📊 Datasets utilisés (tous réels)
Dataset	Source	Patients	Features
PIMA Diabetes	UCI	768	8
Wisconsin Breast Cancer	UCI/sklearn	569	30
Cleveland Heart Disease	UCI	303	13
PCAP Pancreatic Cancer	Kaggle	1 981	4
Norway EHRs Sepsis	Kaggle	110k	3
Chest X-Ray Pneumonia	Kaggle/NIH	5 856	images
🧪 Tests
bash
pytest tests/
🔒 Sécurité implémentée
✅ Validation des inputs utilisateur

✅ Sanitization des noms de fichiers

✅ Chargement sécurisé des modèles

✅ Logging sans données sensibles

✅ Configuration via variables d'environnement

🚀 Déploiement (Streamlit Cloud)
bash
# 1. Pousser le code sur GitHub
git push origin main

# 2. Aller sur https://share.streamlit.io
# 3. Cliquer sur "New app"
# 4. Sélectionner le dépôt
# 5. Main file: dashboard/app.py
# 6. Deploy
📸 Captures d'écran
Interface	Description
🖼️	Dashboard principal avec sélection des pathologies
🖼️	Formulaire de diagnostic (exemple Diabète)
🖼️	Résultat "Risque ÉLEVÉ" (affichage rouge)
🖼️	Résultat "Risque FAIBLE" (affichage vert)
🖼️	Page des performances avec graphiques AUC-ROC
🤝 Contribution
Fork le projet

Créer une branche (git checkout -b feature/amazing)

Committer (git commit -m 'Add amazing feature')

Pousser (git push origin feature/amazing)

Ouvrir une Pull Request

⚠️ Avertissement légal
Ce projet est développé à des fins éducatives et de démonstration uniquement.

❌ Non certifié médicalement

❌ Ne pas utiliser pour des diagnostics réels

❌ Ne remplace pas l'avis d'un médecin

✅ Utilisation uniquement pour apprendre et démontrer

L'auteur décline toute responsabilité en cas d'utilisation médicale réelle.

📄 Licence
Distribué sous licence MIT.



⭐ Mettez une étoile si ce projet vous est utile !

text

---

## 📝 À faire maintenant

```bash
# 1. Créer le fichier
code README.md

# 2. Copier-coller une des deux versions ci-dessus

# 3. Sauvegarder et pousser
git add README.md
git commit -m "Add professional README for GitHub"
git push

