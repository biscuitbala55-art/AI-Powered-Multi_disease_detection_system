# =============================================================================
# AI-Powered Multi-Disease Prediction System
# BSc Computer Systems Engineering - Individual Project
# Diseases: Diabetes | Heart Disease | Kidney Disease
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# =============================================================================
# STEP 1: DATA LOADING
# Place these CSV files in the same folder as this script:
#   - diabetes.csv
#   - heart.csv
#   - kidney_disease.csv
# =============================================================================

def load_datasets():
    datasets = {}

    try:
        df = pd.read_csv('diabetes.csv')
        datasets['diabetes'] = {'data': df, 'target': 'Outcome', 'name': 'Diabetes'}
        print(f"[OK] Diabetes loaded: {df.shape}")
    except FileNotFoundError:
        print("[MISSING] diabetes.csv not found.")

    try:
        df = pd.read_csv('heart.csv')
        datasets['heart'] = {'data': df, 'target': 'target', 'name': 'Heart Disease'}
        print(f"[OK] Heart Disease loaded: {df.shape}")
    except FileNotFoundError:
        print("[MISSING] heart.csv not found.")

    try:
        df = pd.read_csv('kidney_disease.csv')
        datasets['kidney'] = {'data': df, 'target': 'classification', 'name': 'Kidney Disease'}
        print(f"[OK] Kidney Disease loaded: {df.shape}")
    except FileNotFoundError:
        print("[MISSING] kidney_disease.csv not found.")

    return datasets


# =============================================================================
# STEP 2: PREPROCESSING
# =============================================================================

def preprocess(df, target_col, disease_name):
    df = df.copy()

    # Kidney disease label fix
    if disease_name == 'Kidney Disease':
        df[target_col] = df[target_col].str.strip()
        df[target_col] = df[target_col].map({'ckd': 1, 'notckd': 0})

    # Encode any remaining categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col:
            df[col] = le.fit_transform(df[col].astype(str))

    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    feature_names = X.columns.tolist()

    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balance classes with SMOTE
    try:
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    except Exception:
        X_balanced, y_balanced = X_scaled, y.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )

    return X_train, X_test, y_train, y_test, scaler, imputer, feature_names


# =============================================================================
# STEP 3: TRAIN AND COMPARE MODELS
# =============================================================================

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM':                 SVC(probability=True, random_state=42),
        'XGBoost':             XGBClassifier(
                                   n_estimators=100, random_state=42,
                                   use_label_encoder=False, eval_metric='logloss'
                               )
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  Trained: {name}")
    return trained


# =============================================================================
# STEP 4: HYPERPARAMETER TUNING (XGBoost)
# =============================================================================

def tune_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators':  [50, 100, 200],
        'max_depth':     [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample':     [0.8, 1.0]
    }
    xgb = XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    print(f"  Best XGBoost params: {grid.best_params_}")
    return grid.best_estimator_


# =============================================================================
# STEP 5: EVALUATE MODEL
# =============================================================================

def evaluate_model(model, X_test, y_test, disease_name, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_prob)

    print(f"\n  Results for {disease_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix: {disease_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{disease_name.replace(' ', '_')}_confusion_matrix.png", dpi=150)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='steelblue', linewidth=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {disease_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{disease_name.replace(' ', '_')}_roc_curve.png", dpi=150)
    plt.close()

    return {
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1, 'auc': auc
    }


# =============================================================================
# STEP 6: SHAP EXPLAINABILITY
# =============================================================================

def explain_model(model, X_test, feature_names, disease_name, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    try:
        explainer  = shap.Explainer(model, X_test)
        shap_values = explainer(X_test[:100])
        plt.figure()
        shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, show=False)
        plt.title(f'SHAP Feature Importance: {disease_name}')
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/{disease_name.replace(' ', '_')}_shap.png",
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f"  SHAP plot saved for {disease_name}")
    except Exception as e:
        print(f"  SHAP skipped for {disease_name}: {e}")


# =============================================================================
# STEP 7: SAVE MODELS
# =============================================================================

def save_models(best_models, scalers, imputers, feature_map, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    for key, model in best_models.items():
        joblib.dump(model,            f"{output_dir}/{key}_model.pkl")
        joblib.dump(scalers[key],     f"{output_dir}/{key}_scaler.pkl")
        joblib.dump(imputers[key],    f"{output_dir}/{key}_imputer.pkl")
        joblib.dump(feature_map[key], f"{output_dir}/{key}_features.pkl")
        print(f"  Saved: {key}")


# =============================================================================
# STEP 8: MAIN PIPELINE
# =============================================================================

def run_pipeline():
    print("=" * 60)
    print("  AI-Powered Multi-Disease Prediction System")
    print("  Diseases: Diabetes | Heart Disease | Kidney Disease")
    print("=" * 60)

    datasets = load_datasets()
    if not datasets:
        print("\nNo datasets found. Place CSV files in the current directory.")
        return

    best_models     = {}
    scalers         = {}
    imputers        = {}
    feature_map     = {}
    results_summary = []

    for key, info in datasets.items():
        disease_name = info['name']
        print(f"\n{'=' * 50}")
        print(f"  Processing: {disease_name}")
        print(f"{'=' * 50}")

        X_train, X_test, y_train, y_test, scaler, imputer, features = preprocess(
            info['data'], info['target'], disease_name
        )

        print("\n  Comparing classifiers...")
        trained = train_models(X_train, y_train)

        # Select best model by cross-validated F1
        best_name, best_model, best_f1 = None, None, 0
        for name, model in trained.items():
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            avg    = scores.mean()
            print(f"  {name} CV F1: {avg:.4f}")
            if avg > best_f1:
                best_f1, best_name, best_model = avg, name, model

        print(f"\n  Best base model: {best_name} (F1={best_f1:.4f})")
        print("  Tuning XGBoost...")
        tuned        = tune_xgboost(X_train, y_train)
        tuned_scores = cross_val_score(tuned, X_train, y_train, cv=5, scoring='f1')

        if tuned_scores.mean() > best_f1:
            final_model = tuned
            print("  Using tuned XGBoost as final model.")
        else:
            final_model = best_model
            print(f"  Using {best_name} as final model.")

        print("\n  Evaluating...")
        metrics = evaluate_model(final_model, X_test, y_test, disease_name)
        results_summary.append({'Disease': disease_name, **metrics})

        print("  Generating SHAP explanations...")
        explain_model(final_model, X_test, features, disease_name)

        best_models[key] = final_model
        scalers[key]     = scaler
        imputers[key]    = imputer
        feature_map[key] = features

    print("\n  Saving all models...")
    save_models(best_models, scalers, imputers, feature_map)

    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame(results_summary).round(4)
    print(summary_df.to_string(index=False))
    os.makedirs('outputs', exist_ok=True)
    summary_df.to_csv('outputs/results_summary.csv', index=False)
    print("\nDone. Models saved in /models | Plots saved in /outputs")


# =============================================================================
# STREAMLIT WEB APPLICATION  (saved as app.py)
# Run with:  streamlit run app.py
# =============================================================================

STREAMLIT_APP = '''
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="medical symbol",
    layout="wide"
)

st.title("AI-Powered Multi-Disease Prediction System")
st.markdown(
    "Enter patient clinical values to receive a machine learning prediction. "
    "This tool is for research purposes only and does not replace clinical diagnosis."
)
st.markdown("---")

DISEASE_CONFIG = {
    "Diabetes": {
        "key": "diabetes",
        "fields": {
            "Pregnancies":              (0,   20,   1),
            "Glucose":                  (0,   300,  120),
            "BloodPressure":            (0,   200,  70),
            "SkinThickness":            (0,   100,  20),
            "Insulin":                  (0,   900,  80),
            "BMI":                      (0.0, 70.0, 25.0),
            "DiabetesPedigreeFunction": (0.0, 3.0,  0.5),
            "Age":                      (1,   120,  35)
        }
    },
    "Heart Disease": {
        "key": "heart",
        "fields": {
            "age":      (20,  100,  50),
            "sex":      (0,   1,    1),
            "cp":       (0,   3,    0),
            "trestbps": (80,  200,  120),
            "chol":     (100, 600,  200),
            "fbs":      (0,   1,    0),
            "restecg":  (0,   2,    0),
            "thalach":  (60,  220,  150),
            "exang":    (0,   1,    0),
            "oldpeak":  (0.0, 7.0,  1.0),
            "slope":    (0,   2,    1),
            "ca":       (0,   4,    0),
            "thal":     (0,   3,    2)
        }
    },
    "Kidney Disease": {
        "key": "kidney",
        "fields": {
            "age":  (1,   120,    45),
            "bp":   (50,  180,    80),
            "bgr":  (50,  500,    120),
            "bu":   (1,   400,    40),
            "sc":   (0.0, 20.0,   1.2),
            "hemo": (3.0, 20.0,   12.0),
            "pcv":  (10,  60,     40),
            "wc":   (2000, 20000, 8000),
            "rc":   (1.0, 10.0,   5.0)
        }
    }
}

col1, col2 = st.columns([1, 2])

with col1:
    selected = st.selectbox("Select Disease to Predict", list(DISEASE_CONFIG.keys()))
    st.markdown("### Patient Clinical Values")
    config      = DISEASE_CONFIG[selected]
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

with col2:
    if predict_btn:
        key          = config["key"]
        model_path   = f"models/{key}_model.pkl"
        scaler_path  = f"models/{key}_scaler.pkl"
        imputer_path = f"models/{key}_imputer.pkl"

        if not os.path.exists(model_path):
            st.error("Model not found. Run disease_prediction_system.py first to train the models.")
        else:
            model   = joblib.load(model_path)
            scaler  = joblib.load(scaler_path)
            imputer = joblib.load(imputer_path)

            input_df      = pd.DataFrame([user_inputs])
            input_imputed = imputer.transform(input_df)
            input_scaled  = scaler.transform(input_imputed)

            prediction  = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            st.markdown("### Prediction Result")
            if prediction == 1:
                st.error(f"POSITIVE: {selected} detected (Confidence: {probability:.1%})")
            else:
                st.success(f"NEGATIVE: No {selected} detected (Confidence: {1 - probability:.1%})")

            st.warning(
                "Disclaimer: This result is generated by a machine learning model "
                "for research purposes only. It must not be used as a substitute for "
                "professional medical advice, diagnosis, or treatment."
            )

            st.markdown("### Confidence Score")
            st.progress(float(probability))
            st.caption(f"Probability of disease: {probability:.2%}")

            try:
                st.markdown("### SHAP Feature Importance")
                explainer  = shap.Explainer(model, input_scaled)
                shap_vals  = explainer(input_scaled)
                fig, ax    = plt.subplots(figsize=(8, 4))
                shap.waterfall_plot(shap_vals[0], show=False)
                st.pyplot(fig)
                plt.close()
            except Exception:
                st.info("SHAP explanation not available for this model.")

st.markdown("---")
st.caption(
    "AI-Powered Multi-Disease Prediction System | "
    "BSc Computer Systems Engineering | Research Prototype"
)
'''


def save_streamlit_app():
    with open('app.py', 'w') as f:
        f.write(STREAMLIT_APP.strip())
    print("\nStreamlit app saved as app.py")
    print("Launch with:  streamlit run app.py")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    run_pipeline()
    save_streamlit_app()
    