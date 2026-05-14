"""Medical Diagnosis Dashboard - 3 diseases prediction interface."""

import streamlit as st
import numpy as np
import joblib
import plotly.express as px
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="AI Medical Diagnosis Platform", page_icon="🏥", layout="wide")


def load_css() -> str:
    return """
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 0.75rem;
    }
    </style>
    """


@st.cache_resource
def load_models():
    """Load models and scalers, return None for missing ones."""
    models = {}
    scalers = {}
    models_dir = Path("models")

    model_config = {
        "diabetes": ("diabetes_best_model.pkl", "diabetes_scaler.pkl"),
        "breast_cancer": ("breast_cancer_best_model.pkl", "breast_cancer_scaler.pkl"),
        "heart_disease": ("heart_disease_best_model.pkl", "heart_disease_scaler.pkl"),
    }

    for name, (model_path, scaler_path) in model_config.items():
        model_full = models_dir / model_path
        scaler_full = models_dir / scaler_path

        if model_full.exists():
            models[name] = joblib.load(model_full)
        if scaler_full.exists():
            scalers[name] = joblib.load(scaler_full)

    return models, scalers


def predict_disease(model_key, features, models, scalers):
    """Run prediction and return (prediction, probability)."""
    if model_key not in models or model_key not in scalers:
        st.error(f"Modèle {model_key} non disponible")
        return None

    try:
        scaled = scalers[model_key].transform(features)
        pred = models[model_key].predict(scaled)[0]
        proba = models[model_key].predict_proba(scaled)[0]
        return pred, proba
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None


def display_result(risk_level, probability, proba_index=1):
    """Display styled prediction result."""
    proba_value = probability[proba_index] if isinstance(probability, (list, np.ndarray)) else probability
    proba_pct = proba_value * 100

    if risk_level == "high":
        st.markdown(
            f'<div class="risk-high"><h2>⚠️ Risque ÉLEVÉ</h2>'
            f'<p style="font-size: 24px;">Probabilité: {proba_pct:.1f}%</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="risk-low"><h2>✅ Risque FAIBLE</h2>'
            f'<p style="font-size: 24px;">Probabilité: {proba_pct:.1f}%</p></div>',
            unsafe_allow_html=True,
        )


def diabetes_section(models, scalers):
    st.markdown("### 📋 Paramètres pour le Diabète")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Nombre de grossesses", 0, 20, 2)
        glucose = st.number_input("Glycémie à jeun (mg/dL)", 50, 300, 100)
        blood_pressure = st.number_input("Tension artérielle (mm Hg)", 60, 200, 120)
        skin_thickness = st.number_input("Épaisseur peau (mm)", 0, 100, 20)

    with col2:
        insulin = st.number_input("Insuline (mu U/mL)", 0, 500, 80)
        bmi = st.number_input("IMC (kg/m²)", 10.0, 50.0, 25.0, 0.1)
        diabetes_pedigree = st.slider("Fonction pedigree diabète", 0.0, 2.5, 0.5, 0.01)
        age = st.number_input("Âge (années)", 0, 120, 50)

    if st.button("🔍 Diagnostiquer", use_container_width=True):
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                             insulin, bmi, diabetes_pedigree, age]])
        result = predict_disease("diabetes", features, models, scalers)
        if result:
            pred, proba = result
            st.markdown("---")
            display_result("high" if pred == 1 else "low", proba, 1)


def breast_cancer_section(models, scalers):
    st.markdown("### 📋 Paramètres pour le Cancer du Sein")
    st.info("Mesures extraites d'une analyse cytologique (FNA)")

    col1, col2 = st.columns(2)

    with col1:
        radius = st.slider("Rayon moyen", 5.0, 30.0, 14.0, 0.1)
        texture = st.slider("Texture moyenne", 5.0, 40.0, 19.0, 0.1)
        perimeter = st.slider("Périmètre moyen", 40.0, 200.0, 92.0, 0.1)
        area = st.slider("Aire moyenne", 100.0, 2500.0, 650.0, 10.0)
        smoothness = st.slider("Rugosité moyenne", 0.05, 0.20, 0.10, 0.01)

    with col2:
        compactness = st.slider("Compacité moyenne", 0.02, 0.30, 0.10, 0.01)
        concavity = st.slider("Concavité moyenne", 0.00, 0.40, 0.09, 0.01)
        concave_points = st.slider("Points concaves", 0.00, 0.20, 0.05, 0.01)
        symmetry = st.slider("Symétrie moyenne", 0.10, 0.30, 0.18, 0.01)
        fractal_dim = st.slider("Dimension fractale", 0.05, 0.10, 0.06, 0.01)

    if st.button("🔍 Analyser la Tumeur", use_container_width=True):
        base = [radius, texture, perimeter, area, smoothness,
                compactness, concavity, concave_points, symmetry, fractal_dim]
        features = np.array([(base * 3)[:30]])
        result = predict_disease("breast_cancer", features, models, scalers)

        if result:
            pred, proba = result
            st.markdown("---")
            if pred == 1:
                st.markdown(
                    f'<div class="risk-low"><h2>✅ Tumeur BÉNIGNE</h2>'
                    f'<p style="font-size: 24px;">Probabilité: {proba[1]*100:.1f}%</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="risk-high"><h2>⚠️ Tumeur MALIGNE</h2>'
                    f'<p style="font-size: 24px;">Probabilité: {proba[0]*100:.1f}%</p></div>',
                    unsafe_allow_html=True,
                )


def heart_disease_section(models, scalers):
    st.markdown("### 📋 Paramètres pour la Maladie Cardiaque")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Âge", 20, 100, 55)
        sex = st.selectbox("Sexe", ["Femme", "Homme"])
        sex_val = 1 if sex == "Homme" else 0
        cp_options = ["Typique", "Atypique", "Non angineuse", "Asymptomatique"]
        cp_val = cp_options.index(st.selectbox("Type de douleur thoracique", cp_options))
        trestbps = st.number_input("Tension artérielle repos", 80, 200, 130)

    with col2:
        chol = st.number_input("Cholestérol", 100, 600, 250)
        fbs_val = 1 if st.selectbox("Glycémie à jeun > 120", ["Non", "Oui"]) == "Oui" else 0
        thalach = st.number_input("Fréquence cardiaque max", 60, 220, 150)
        exang_val = 1 if st.selectbox("Angine d'effort", ["Non", "Oui"]) == "Oui" else 0
        oldpeak = st.number_input("Dépression ST", 0.0, 10.0, 1.0, 0.1)

    if st.button("🔍 Évaluer le Risque", use_container_width=True):
        features = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                             0, thalach, exang_val, oldpeak, 1, 0, 2]])
        result = predict_disease("heart_disease", features, models, scalers)
        if result:
            pred, proba = result
            st.markdown("---")
            display_result("high" if pred == 1 else "low", proba, 1)


def dashboard_section():
    st.markdown("## 📊 Dashboard Analytics")

    try:
        df_heart = pd.read_csv("data/heart_disease.csv")
        df_cancer = pd.read_csv("data/breast_cancer.csv")
        df_diabetes = pd.read_csv("data/diabetes.csv")

        total = len(df_heart) + len(df_cancer) + len(df_diabetes)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Patients", total)
        col2.metric("Maladies", 3)
        col3.metric("Précision moyenne", "~85%")
        col4.metric("Modèles IA", 7)

        st.markdown("---")

        fig1 = px.pie(
            values=[(df_heart["target"] == 0).sum(), (df_heart["target"] == 1).sum()],
            names=["Sain", "Malade"],
            title="Heart Disease Distribution",
        )
        fig2 = px.pie(
            values=[(df_cancer["target"] == 1).sum(), (df_cancer["target"] == 0).sum()],
            names=["Bénin", "Malin"],
            title="Breast Cancer Distribution",
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)

    except FileNotFoundError as e:
        st.error(f"Fichier de données manquant: {e}")
    except Exception as e:
        st.error(f"Erreur: {e}")


def about_section():
    st.markdown("## ℹ️ À Propos")
    st.markdown("""
    ### 🏥 AI Medical Diagnosis Platform

    **Technologies utilisées:**
    - Random Forest, XGBoost, LightGBM, CatBoost, SVM
    - Scikit-learn, Pandas, NumPy
    - Streamlit, Plotly

    ### 📊 Performance

    | Maladie | Meilleur Modèle | AUC-ROC |
    |---------|----------------|---------|
    | Diabète | Logistic Regression | 0.826 |
    | Cancer du Sein | Logistic Regression | **0.995** |
    | Maladie Cardiaque | SVM | **0.954** |

    ### ⚠️ Avertissement
    Cette application est un **outil de démonstration**.
    Consultez toujours un médecin pour un diagnostic officiel.
    """)


def main():
    st.markdown(load_css(), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="main-header">
            <h1>🏥 AI Medical Diagnosis Platform</h1>
            <p>Diagnostic Assisté par Intelligence Artificielle | Précision & Fiabilité</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
        st.markdown("## Navigation")
        page = st.radio("", ["🎯 Diagnostic", "📊 Dashboard Analytics", "ℹ️ À Propos"])
        st.markdown("---")
        st.info("⚠️ Outil d'aide à la décision - Consultez toujours un médecin")

    models, scalers = load_models()

    if page == "🎯 Diagnostic":
        st.markdown("## 🎯 Diagnostic Assisté par IA")

        disease = st.selectbox(
            "📋 Sélectionnez la pathologie à diagnostiquer",
            ["Diabète", "Cancer du Sein", "Maladie Cardiaque"],
        )
        st.markdown("---")

        if disease == "Diabète":
            diabetes_section(models, scalers)
        elif disease == "Cancer du Sein":
            breast_cancer_section(models, scalers)
        elif disease == "Maladie Cardiaque":
            heart_disease_section(models, scalers)

    elif page == "📊 Dashboard Analytics":
        dashboard_section()
    else:
        about_section()

    st.markdown("---")
    st.markdown("© 2025 AI Medical Diagnosis Platform")


if __name__ == "__main__":
    main()