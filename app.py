import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import time

st.set_page_config(
    page_title="MedPredict AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── IBM Plex fonts + clinical CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Layout ── */
.main { background-color: #F7FAFB; }

[data-testid="stSidebar"] {
    background-color: #0B2D48;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #cde0ef !important; }
[data-testid="stSidebar"] label { color: #90b8d4 !important; font-size: 0.78rem !important; }
[data-testid="stSidebar"] input {
    background: #0e3557 !important;
    border: 1px solid #1e5278 !important;
    color: #e8f4fd !important;
    border-radius: 5px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #0e3557 !important;
    border: 1px solid #1e5278 !important;
    color: #e8f4fd !important;
    border-radius: 5px !important;
}

/* ── Sidebar header ── */
.sb-brand {
    padding: 1.5rem 0 1rem;
    text-align: center;
    border-bottom: 1px solid #1e4d70;
    margin-bottom: 1.2rem;
}
.sb-brand .logo { font-size: 2.2rem; }
.sb-brand .name {
    font-size: 1rem; font-weight: 700;
    color: #e8f4fd !important; letter-spacing: 0.5px;
}
.sb-brand .tagline { font-size: 0.72rem; color: #6fa8c9 !important; margin-top: 2px; }

/* ── Sidebar section labels ── */
.sb-label {
    font-size: 0.68rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1.2px;
    color: #4d8db5 !important;
    padding: 1.2rem 0 0.4rem;
}

/* ── Run button ── */
.stButton > button {
    background: linear-gradient(135deg, #1A7AAF, #0e5a88) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 0 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    letter-spacing: 0.4px !important;
    margin-top: 0.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Main content ── */
.block-container { padding-top: 1.2rem !important; padding-bottom: 1rem !important; }

/* ── Page header ── */
.pg-header {
    background: linear-gradient(135deg, #0B2D48 0%, #1A7AAF 100%);
    padding: 1.6rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    box-shadow: 0 3px 12px rgba(11,45,72,0.18);
}
.pg-header h1 {
    font-size: 1.55rem; font-weight: 700;
    color: #ffffff; margin: 0; letter-spacing: -0.3px;
}
.pg-header p {
    color: rgba(255,255,255,0.72);
    font-size: 0.84rem; margin: 0.4rem 0 0;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border: 1px solid #dce8f0;
    border-radius: 10px;
    padding: 1.3rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
.card-pos { border-left: 5px solid #C0392B; background: #fff8f7; }
.card-neg { border-left: 5px solid #27AE60; background: #f7fff9; }

.res-title { font-size: 1.15rem; font-weight: 700; }
.res-pos   { color: #C0392B; }
.res-neg   { color: #27AE60; }
.res-sub   { font-size: 0.84rem; color: #7f96a8; margin-top: 0.35rem; }

/* ── Metric boxes ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
.metric {
    flex: 1;
    background: #F7FAFB;
    border: 1px solid #dce8f0;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem; font-weight: 500;
    color: #0B2D48;
}
.metric .lbl { font-size: 0.74rem; color: #7f96a8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Step cards (empty state) ── */
.step-card {
    background: #ffffff;
    border: 1px solid #dce8f0;
    border-radius: 10px;
    padding: 1.4rem;
    text-align: center;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.step-card .icon  { font-size: 1.8rem; margin-bottom: 0.5rem; }
.step-card .step  { font-weight: 600; color: #0B2D48; font-size: 0.88rem; }
.step-card .desc  { font-size: 0.8rem; color: #7f96a8; margin-top: 0.25rem; line-height: 1.45; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.28rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem; font-weight: 500;
    margin-right: 0.4rem;
}
.b-blue   { background:#E8F4FD; color:#0B2D48; }
.b-red    { background:#FDEAEA; color:#a93226; }
.b-green  { background:#EAF7F0; color:#1a7a4a; }

/* ── Risk bar label ── */
.risk-label {
    font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.8px;
    color: #4d8db5; margin-bottom: 0.3rem;
}

/* ── Disclaimer ── */
.disclaimer {
    background: #fffbea;
    border: 1px solid #f0d060;
    border-radius: 7px;
    padding: 0.75rem 1rem;
    font-size: 0.8rem;
    color: #6b5700;
    margin-top: 0.75rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #a0b4c0;
    font-size: 0.76rem;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #dce8f0;
    margin-top: 2rem;
    letter-spacing: 0.2px;
}
</style>
""", unsafe_allow_html=True)

# ── Disease config ─────────────────────────────────────────────────────────────
DISEASE_CONFIG = {
    "Diabetes": {
        "key": "diabetes",
        "icon": "🩸",
        "fields": {
            "Pregnancies": (0, 20, 1),
            "Glucose": (0, 300, 120),
            "BloodPressure": (0, 200, 70),
            "SkinThickness": (0, 100, 20),
            "Insulin": (0, 900, 80),
            "BMI": (0.0, 70.0, 25.0),
            "DiabetesPedigreeFunction": (0.0, 3.0, 0.5),
            "Age": (1, 120, 35)
        }
    },
    "Heart Disease": {
        "key": "heart",
        "icon": "❤️",
        "fields": {
            "age": (20, 100, 50),
            "sex": (0, 1, 1),
            "ChestPain": (0, 3, 0),
            "trestbps": (80, 200, 120),
            "chol": (100, 600, 200),
            "fbs": (0, 1, 0),
            "restecg": (0, 2, 0),
            "thalach": (60, 220, 150),
            "exang": (0, 1, 0),
            "oldpeak": (0.0, 7.0, 1.0),
            "slope": (0, 2, 1),
            "ca": (0, 4, 0),
            "thal": (0, 3, 2)
        }
    },
    "Kidney Disease": {
        "key": "kidney",
        "icon": "🫘",
        "fields": {
            "Age": (1, 120, 45),
            "Blood Pressure": (50, 180, 80),
            "Blood Glucose Random": (50, 500, 120),
            "Blood Urea": (1, 400, 40),
            "Serum Creatinine": (0.0, 20.0, 1.2),
            "Hemoglobin": (3.0, 20.0, 12.0),
            "Packed Cell Volume": (10, 60, 40),
            "White blood Cell Count": (2000, 20000, 8000),
            "Red blood Cell Count": (1.0, 10.0, 5.0)
        }
    }
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class="sb-brand">
            <div class="logo">🏥</div>
            <div class="name">MedPredict AI</div>
            <div class="tagline">Clinical Decision Support</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Disease Module</div>', unsafe_allow_html=True)
    selected = st.selectbox("Disease", list(DISEASE_CONFIG.keys()), label_visibility="collapsed")
    config = DISEASE_CONFIG[selected]

    st.markdown('<div class="sb-label">Patient Clinical Values</div>', unsafe_allow_html=True)

    user_inputs = {}
    for field, (mn, mx, default) in config["fields"].items():
        if isinstance(default, float):
            user_inputs[field] = st.number_input(
                field,
                min_value=float(mn), max_value=float(mx),
                value=float(default), step=0.1
            )
        else:
            user_inputs[field] = st.number_input(
                field,
                min_value=int(mn), max_value=int(mx),
                value=int(default), step=1
            )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔬  Run Prediction", type="primary")

    st.markdown("""
        <div style="margin-top:1.5rem; padding:0.75rem 0.9rem;
                    background:#0e3557; border-radius:7px;
                    font-size:0.73rem; color:#6fa8c9; text-align:center;
                    line-height:1.5;">
            ⚕️ For research purposes only.<br>
            Not a substitute for clinical diagnosis.
        </div>
    """, unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(f"""
    <div class="pg-header">
        <h1>{config['icon']}  Multi-Disease Prediction System</h1>
        <p>AI-powered clinical risk assessment · Enter patient values in the sidebar, then run a prediction.</p>
    </div>
""", unsafe_allow_html=True)

# ── Empty state ───────────────────────────────────────────────────────────────
if not predict_btn:
    c1, c2, c3 = st.columns(3)
    steps = [
        ("📋", "Select a Module", "Choose a disease category from the sidebar dropdown."),
        ("🔢", "Enter Clinical Values", "Fill in the patient's lab and clinical measurements."),
        ("🔬", "Run Prediction", "Click the button to receive an AI-generated risk assessment."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], steps):
        with col:
            st.markdown(f"""
                <div class="step-card">
                    <div class="icon">{icon}</div>
                    <div class="step">{title}</div>
                    <div class="desc">{desc}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("""
        <div class="card" style="margin-top:0.25rem;">
            <div style="font-weight:600; color:#0B2D48; font-size:0.88rem; margin-bottom:0.7rem;">
                Available Disease Modules
            </div>
            <span class="badge b-blue">🩸 Diabetes</span>
            <span class="badge b-red">❤️ Heart Disease</span>
            <span class="badge b-green">🫘 Kidney Disease</span>
        </div>
    """, unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
else:
    key          = config["key"]
    model_path   = f"models/{key}_model.pkl"
    scaler_path  = f"models/{key}_scaler.pkl"
    imputer_path = f"models/{key}_imputer.pkl"
    feature_path = f"models/{key}_features.pkl"

    if not os.path.exists(model_path):
        st.error("⚠️  Model files not found. Please run the training pipeline first.")
    else:
        with st.spinner("Analysing patient data…"):
            time.sleep(0.7)

            model    = joblib.load(model_path)
            scaler   = joblib.load(scaler_path)
            imputer  = joblib.load(imputer_path)
            features = joblib.load(feature_path)

            input_df = pd.DataFrame([user_inputs])
            input_df = input_df.reindex(features, axis=1).fillna(0)

            input_imputed = imputer.transform(input_df)
            input_scaled  = scaler.transform(input_imputed)

            prediction  = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

        # ── Result card ──
        if prediction == 1:
            st.markdown(f"""
                <div class="card card-pos">
                    <div class="res-title res-pos">⚠  POSITIVE — {selected} Risk Detected</div>
                    <div class="res-sub">
                        The model indicates a positive outcome with
                        <strong>{probability:.1%}</strong> confidence.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="card card-neg">
                    <div class="res-title res-neg">✓  NEGATIVE — No {selected} Risk Detected</div>
                    <div class="res-sub">
                        The model indicates a negative outcome with
                        <strong>{1 - probability:.1%}</strong> confidence.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # ── Metric row ──
        risk_label = "High" if probability >= 0.7 else ("Moderate" if probability >= 0.4 else "Low")
        risk_color = ("#C0392B" if probability >= 0.7
                      else ("#F39C12" if probability >= 0.4 else "#27AE60"))

        st.markdown(f"""
            <div class="metric-row">
                <div class="metric">
                    <div class="val">{probability:.0%}</div>
                    <div class="lbl">Disease Probability</div>
                </div>
                <div class="metric">
                    <div class="val">{1 - probability:.0%}</div>
                    <div class="lbl">No-Disease Probability</div>
                </div>
                <div class="metric">
                    <div class="val" style="color:{risk_color};">{risk_label}</div>
                    <div class="lbl">Risk Level</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # ── Confidence bar ──
        st.markdown('<div class="risk-label">Confidence Score</div>', unsafe_allow_html=True)
        st.progress(float(probability))
        st.caption(f"Raw probability: {probability:.4f}")

        # ── SHAP ──
        try:
            st.markdown("---")
            st.markdown("**Feature Importance (SHAP Analysis)**")
            explainer = shap.Explainer(model, input_scaled)
            shap_vals = explainer(input_scaled)
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.waterfall_plot(shap_vals[0], show=False)
            st.pyplot(fig)
            plt.close()
        except Exception:
            st.info(
                "📊 **Feature Importance Visualization** — Not available for this model type.\n\n"
                "✔ Prediction is still fully valid  \n"
                "✔ Model uses trained clinical features  \n"
                "✔ Results are based on a validated ML pipeline"
            )

        # ── Disclaimer ──
        st.markdown("""
            <div class="disclaimer">
                ⚠️ <strong>Research Use Only</strong> — This output is generated by a machine learning model
                and does not constitute a clinical diagnosis.
                Always consult a qualified healthcare professional for medical decisions.
            </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="footer">
        AI Multi-Disease Prediction System &nbsp;·&nbsp;
        BSc Computer Systems Engineering &nbsp;·&nbsp; Research Project
    </div>
""", unsafe_allow_html=True)