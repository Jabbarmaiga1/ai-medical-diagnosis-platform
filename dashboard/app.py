"""Medical Diagnosis Dashboard - 6 diseases prediction interface."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path
import joblib
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Medical Diagnosis AI", page_icon="🏥", layout="wide")


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
    """Load all ML models."""
    models = {}
    scalers = {}

    model_files = {
        "diabetes": "models/diabetes_best_model.pkl",
        "breast_cancer": "models/breast_cancer_best_model.pkl",
        "heart_disease": "models/heart_disease_best_model.pkl",
        "pancreatic_cancer": "models/pancreatic_cancer_model.pkl",
        "sepsis": "models/sepsis_model_improved.pkl",
    }

    scaler_files = {
        "diabetes": "models/diabetes_scaler.pkl",
        "breast_cancer": "models/breast_cancer_scaler.pkl",
        "heart_disease": "models/heart_disease_scaler.pkl",
        "pancreatic_cancer": "models/pancreatic_cancer_scaler.pkl",
        "sepsis": "models/sepsis_scaler_improved.pkl",
    }

    for name, path in model_files.items():
        p = Path(path)
        if p.exists():
            models[name] = joblib.load(p)

    for name, path in scaler_files.items():
        p = Path(path)
        if p.exists():
            scalers[name] = joblib.load(p)

    # Load CNN model for pneumonia
    cnn_path = Path("models/pneumonia_cnn_model.h5")
    cnn_model = None
    if cnn_path.exists():
        try:
            cnn_model = tf.keras.models.load_model(cnn_path)
        except Exception as e:
            st.warning(f"CNN model not loaded: {e}")

    return models, scalers, cnn_model


def display_result(risk_level: str, probability: float):
    if risk_level == "high":
        st.markdown(
            f'<div class="risk-high"><h2>⚠️ Risque ÉLEVÉ</h2>'
            f'<p style="font-size: 24px;">Probabilité: {probability*100:.1f}%</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="risk-low"><h2>✅ Risque FAIBLE</h2>'
            f'<p style="font-size: 24px;">Probabilité: {probability*100:.1f}%</p></div>',
            unsafe_allow_html=True,
        )


def diabetes_form(models, scalers):
    st.markdown("### Paramètres du patient")
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Grossesses", 0, 20, 2)
        glucose = st.number_input("Glycémie (mg/dL)", 50, 300, 100)
        bp = st.number_input("Tension artérielle (mm Hg)", 60, 200, 120)
        skin = st.number_input("Épaisseur peau (mm)", 0, 100, 20)
    with col2:
        insulin = st.number_input("Insuline (mu U/mL)", 0, 500, 80)
        bmi = st.number_input("IMC (kg/m²)", 10.0, 50.0, 25.0, 0.1)
        pedigree = st.slider("Antécédents familiaux", 0.0, 2.5, 0.5, 0.01)
        age = st.number_input("Âge", 0, 120, 50)

    if st.button("Diagnostiquer"):
        features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]])
        if "diabetes" in scalers:
            scaled = scalers["diabetes"].transform(features)
            proba = models["diabetes"].predict_proba(scaled)[0][1]
            pred = models["diabetes"].predict(scaled)[0]
            display_result("high" if pred == 1 else "low", proba)


def breast_cancer_form(models, scalers):
    st.markdown("### Caractéristiques de la tumeur")
    col1, col2 = st.columns(2)
    with col1:
        radius = st.slider("Rayon", 5.0, 30.0, 14.0, 0.1)
        texture = st.slider("Texture", 5.0, 40.0, 19.0, 0.1)
        perimeter = st.slider("Périmètre", 40.0, 200.0, 92.0, 0.1)
        area = st.slider("Aire", 100.0, 2500.0, 650.0, 10.0)
    with col2:
        smoothness = st.slider("Rugosité", 0.05, 0.20, 0.10, 0.01)
        compactness = st.slider("Compacité", 0.02, 0.30, 0.10, 0.01)
        concavity = st.slider("Concavité", 0.00, 0.40, 0.09, 0.01)
        symmetry = st.slider("Symétrie", 0.10, 0.30, 0.18, 0.01)

    if st.button("Analyser"):
        base = [radius, texture, perimeter, area, smoothness, compactness, concavity, 0.05, symmetry, 0.06]
        features = np.array([(base * 3)[:30]])
        if "breast_cancer" in scalers:
            scaled = scalers["breast_cancer"].transform(features)
            proba = models["breast_cancer"].predict_proba(scaled)[0]
            pred = models["breast_cancer"].predict(scaled)[0]
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


def heart_disease_form(models, scalers):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Âge", 20, 100, 55)
        sex = 1 if st.selectbox("Sexe", ["Femme", "Homme"]) == "Homme" else 0
        cp_options = ["Typique", "Atypique", "Non angineuse", "Asymptomatique"]
        cp_val = cp_options.index(st.selectbox("Douleur thoracique", cp_options))
    with col2:
        trestbps = st.number_input("Tension repos (mm Hg)", 80, 200, 130)
        chol = st.number_input("Cholestérol (mg/dL)", 100, 600, 250)
        thalach = st.number_input("FC max (bpm)", 60, 220, 150)

    if st.button("Évaluer"):
        features = np.array([[age, sex, cp_val, trestbps, chol, 0, 0, thalach, 0, 1.0, 1, 0, 2]])
        if "heart_disease" in scalers:
            scaled = scalers["heart_disease"].transform(features)
            proba = models["heart_disease"].predict_proba(scaled)[0][1]
            pred = models["heart_disease"].predict(scaled)[0]
            display_result("high" if pred == 1 else "low", proba)


def pancreatic_cancer_form(models, scalers):
    col1, col2 = st.columns(2)
    with col1:
        cancer_site = st.selectbox("Code cancer", [21100, 21110, 21120, 21130])
        age_group = st.selectbox("Groupe âge", list(range(1, 10)))
    with col2:
        sex_code = st.selectbox("Sexe", [1, 2], format_func=lambda x: "Homme" if x == 1 else "Femme")
        race_code = st.selectbox("Race", [2106, 2054, 2000, 2500])

    if st.button("Évaluer risque"):
        features = np.array([[cancer_site, age_group, sex_code, race_code]])
        if "pancreatic_cancer" in scalers:
            scaled = scalers["pancreatic_cancer"].transform(features)
            proba = models["pancreatic_cancer"].predict_proba(scaled)[0][1]
            pred = models["pancreatic_cancer"].predict(scaled)[0]
            display_result("high" if pred == 1 else "low", proba)


def sepsis_form(models, scalers):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Âge", 0, 120, 60)
        sex_val = 1 if st.selectbox("Sexe", ["Femme", "Homme"]) == "Homme" else 0
    with col2:
        los = st.number_input("Durée séjour (jours)", 0, 365, 10)

    if st.button("Évaluer risque Sepsis"):
        features = np.array([[age, sex_val, los]])
        if "sepsis" in scalers:
            scaled = scalers["sepsis"].transform(features)
            proba = models["sepsis"].predict_proba(scaled)[0][1]
            pred = models["sepsis"].predict(scaled)[0]
            if pred == 1:
                st.markdown(
                    f'<div class="risk-high"><h2>🚨 Risque ÉLEVÉ de Sepsis</h2>'
                    f'<p style="font-size: 24px;">Probabilité: {proba*100:.1f}%</p>'
                    f'<p>Consultation urgente recommandée</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                display_result("low", proba)


def pneumonia_form(cnn_model):
    st.markdown("### Radiographie thoracique")
    st.info("Téléchargez une radio pulmonaire pour détection de pneumonie")

    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and cnn_model is not None:
        from PIL import Image
        img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
        st.image(img, caption="Radiographie téléchargée", width=300)

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyse en cours..."):
            proba = float(cnn_model.predict(img_array)[0][0])

        if proba > 0.5:
            st.markdown(
                f'<div class="risk-high"><h2>⚠️ PNEUMONIE DÉTECTÉE</h2>'
                f'<p style="font-size: 24px;">Probabilité: {proba*100:.1f}%</p>'
                f'<p>Consultation médicale recommandée</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="risk-low"><h2>✅ RADIOGRAPHIE NORMALE</h2>'
                f'<p style="font-size: 24px;">Probabilité: {(1-proba)*100:.1f}%</p></div>',
                unsafe_allow_html=True,
            )
    elif uploaded_file is not None and cnn_model is None:
        st.error("Modèle Pneumonie non disponible")


def performance_page():
    st.markdown("## 📊 Performance des Modèles")
    data = {
        "Maladie": ["Diabète", "Cancer Sein", "Cardiaque", "Cancer Pancréas", "Sepsis", "Pneumonie"],
        "AUC-ROC": [0.826, 0.995, 0.954, 0.852, 0.735, 0.94],
        "Accuracy": [0.78, 0.97, 0.90, 0.78, 0.63, 0.915],
    }
    df = pd.DataFrame(data)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df, x="Maladie", y="AUC-ROC", title="AUC-ROC par maladie", color="AUC-ROC")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(df, x="Maladie", y="Accuracy", title="Accuracy par maladie", color="Accuracy")
        st.plotly_chart(fig, use_container_width=True)


def about_page():
    st.markdown("## ℹ️ À Propos")
    st.markdown("""
    ### AI Medical Diagnosis Platform

    **6 pathologies détectables :**
    - Diabète (AUC-ROC: 0.826)
    - Cancer du Sein (AUC-ROC: 0.995)
    - Maladie Cardiaque (AUC-ROC: 0.954)
    - Cancer du Pancréas (AUC-ROC: 0.852)
    - Sepsis (AUC-ROC: 0.735)
    - Pneumonie (Accuracy: 91.5%)

    **Technologies :** Python, Scikit-learn, XGBoost, TensorFlow, Streamlit

    ⚠️ **Avertissement :** Outil de démonstration - Ne remplace pas un diagnostic médical
    """)


def main():
    st.markdown(load_css(), unsafe_allow_html=True)
    st.markdown(
        '<div class="main-header"><h1>🏥 AI Medical Diagnosis Platform</h1>'
        '<p>6 pathologies | Intelligence Artificielle | Précision clinique</p></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
        page = st.radio("Navigation", ["🎯 Diagnostic", "📊 Performance", "ℹ️ À Propos"])
        st.info("⚠️ Outil d'aide à la décision - Consultez toujours un médecin")

    models, scalers, cnn_model = load_models()

    if page == "🎯 Diagnostic":
        st.markdown("## 🎯 Diagnostic Assisté par IA")
        disease = st.selectbox(
            "Pathologie",
            ["Diabète", "Cancer du Sein", "Maladie Cardiaque", "Cancer Pancréas", "Sepsis", "Pneumonie"],
        )
        st.markdown("---")

        if disease == "Diabète":
            diabetes_form(models, scalers)
        elif disease == "Cancer du Sein":
            breast_cancer_form(models, scalers)
        elif disease == "Maladie Cardiaque":
            heart_disease_form(models, scalers)
        elif disease == "Cancer Pancréas":
            pancreatic_cancer_form(models, scalers)
        elif disease == "Sepsis":
            sepsis_form(models, scalers)
        elif disease == "Pneumonie":
            pneumonia_form(cnn_model)

    elif page == "📊 Performance":
        performance_page()
    else:
        about_page()


if __name__ == "__main__":
    main()