# 

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.inspection import permutation_importance

st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

st.title("AI-Powered Multi-Disease Prediction System")
st.markdown("Enter patient clinical values to receive a prediction (Research only).")
st.markdown("---")

# ---------------- CONFIG ----------------
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
    selected = st.selectbox("Select Disease", list(DISEASE_CONFIG.keys()))
    config = DISEASE_CONFIG[selected]

    st.markdown("### Patient Input")

    user_inputs = {}
    for field, (mn, mx, default) in config["fields"].items():
        if isinstance(default, float):
            user_inputs[field] = st.number_input(field, float(mn), float(mx), float(default), 0.1)
        else:
            user_inputs[field] = st.number_input(field, int(mn), int(mx), int(default), 1)

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
            st.error("Model not found. Please train first.")
        else:

            model   = joblib.load(model_path)
            scaler  = joblib.load(scaler_path)
            imputer = joblib.load(imputer_path)
            features = joblib.load(feature_path)

            # ---------------- FIX FEATURE MISMATCH ----------------
            input_df = pd.DataFrame([user_inputs])
            input_df = input_df.reindex(features, axis=1)
            input_df = input_df.fillna(0)

            input_imputed = imputer.transform(input_df)
            input_scaled  = scaler.transform(input_imputed)

            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            # ---------------- RESULT ----------------
            st.markdown("### Prediction Result")

            if prediction == 1:
                st.error(f"POSITIVE - {selected} detected ({probability:.1%})")
            else:
                st.success(f"NEGATIVE - No {selected} detected ({1-probability:.1%})")

            st.progress(float(probability))
            st.caption(f"Risk Probability: {probability:.2%}")

            # ---------------- FEATURE IMPORTANCE ----------------
            st.markdown("### Feature Importance")

            try:
                result = permutation_importance(
                    model,
                    input_scaled,
                    model.predict(input_scaled),
                    n_repeats=10,
                    random_state=42
                )

                importance = result.importances_mean

                fi_df = pd.DataFrame({
                    "Feature": input_df.columns,
                    "Importance": importance
                }).sort_values("Importance")

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(fi_df["Feature"], fi_df["Importance"], color="steelblue")
                ax.set_title("Permutation Feature Importance")

                st.pyplot(fig)

            except Exception:
                st.info("Feature importance not available for this model.")

st.markdown("---")
st.caption("AI Multi-Disease Prediction System | BSc Project")