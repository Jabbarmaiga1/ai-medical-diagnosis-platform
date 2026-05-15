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

