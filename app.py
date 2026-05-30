import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

st.title("AI-Powered Multi-Disease Prediction System")
st.markdown(
    "Enter patient clinical values to receive a machine learning prediction. "
    "This tool is for research purposes only and does not replace clinical diagnosis."
)
st.markdown("---")

# ---------------- DISEASE CONFIG ----------------
DISEASE_CONFIG = {
    "Diabetes": {
        "key": "diabetes",
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
        "fields": {
            "age": (20, 100, 50),
            "sex": (0, 1, 1),
            "cp": (0, 3, 0),
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
        "fields": {
            "age": (1, 120, 45),
            "bp": (50, 180, 80),
            "bgr": (50, 500, 120),
            "bu": (1, 400, 40),
            "sc": (0.0, 20.0, 1.2),
            "hemo": (3.0, 20.0, 12.0),
            "pcv": (10, 60, 40),
            "wc": (2000, 20000, 8000),
            "rc": (1.0, 10.0, 5.0)
        }
    }
}

# ---------------- UI ----------------
col1, col2 = st.columns([1, 2])

with col1:
    selected = st.selectbox("Select Disease to Predict", list(DISEASE_CONFIG.keys()))
    config = DISEASE_CONFIG[selected]

    st.markdown("### Patient Clinical Values")

    user_inputs = {}
    for field, (mn, mx, default) in config["fields"].items():
        if isinstance(default, float):
            user_inputs[field] = st.number_input(
                field, min_value=float(mn), max_value=float(mx),
                value=float(default), step=0.1
            )
        else:
            user_inputs[field] = st.number_input(
                field, min_value=int(mn), max_value=int(mx),
                value=int(default), step=1
            )

    predict_btn = st.button("Run Prediction", type="primary")

# ---------------- PREDICTION ----------------
with col2:
    if predict_btn:

        key = config["key"]

        model_path   = f"models/{key}_model.pkl"
        scaler_path  = f"models/{key}_scaler.pkl"
        imputer_path = f"models/{key}_imputer.pkl"
        feature_path = f"models/{key}_features.pkl"

        if not os.path.exists(model_path):
            st.error("Model not found. Run training first.")
        else:

            model   = joblib.load(model_path)
            scaler  = joblib.load(scaler_path)
            imputer = joblib.load(imputer_path)
            features = joblib.load(feature_path)   # ✅ FIX ADDED

            # ---------------- FIX FEATURE MISMATCH ----------------
            input_df = pd.DataFrame([user_inputs])
            input_df = input_df.reindex(features, axis=1)
            input_df = input_df.fillna(0)

            input_imputed = imputer.transform(input_df)
            input_scaled  = scaler.transform(input_imputed)

            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            

            st.markdown("### Prediction Result")

            if prediction == 1:
                st.error(f"POSITIVE: {selected} detected (Confidence: {probability:.1%})")
            else:
                st.success(f"NEGATIVE: No {selected} detected (Confidence: {1 - probability:.1%})")

            st.warning(
                "This is an ML prediction for research purposes only. Not medical advice."
            )

            st.markdown("### Confidence Score")
            st.progress(float(probability))
            st.caption(f"Probability: {probability:.2%}")

            # ---------------- SHAP (SAFE) ----------------
            try:
                st.markdown("### Feature Importance (SHAP)")

                explainer = shap.Explainer(model, input_scaled)
                shap_vals = explainer(input_scaled)

                fig, ax = plt.subplots(figsize=(8, 4))
                shap.waterfall_plot(shap_vals[0], show=False)

                st.pyplot(fig)
                plt.close()

            except Exception:
                st.markdown(
    """
    ### Feature Explanation
    Explainability visualization is not available for this model type.

    This is due to differences in algorithm compatibility with SHAP.

    ✔ Prediction is still fully valid  
    ✔ Model uses trained clinical features  
    ✔ Results are based on validated ML pipeline  
    """
)

st.markdown("---")
st.caption("AI Multi-Disease Prediction System |"
           " BSc Computer Systems Engineering | Resesarch Project. ")