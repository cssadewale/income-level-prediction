"""
Income Level Prediction — Streamlit Web Application
====================================================
Author  : Adewale Adeagbo
GitHub  : https://github.com/cssadewale
LinkedIn: https://linkedin.com/in/adewalesamsonadeagbo

This app loads the trained Random Forest model and scaler, accepts
user-input census features, and returns a predicted income class
with a probability score.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Income Level Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Load Model and Scaler ─────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load the saved model and scaler. Cached so they load only once."""
    model_path  = os.path.join("models", "income_prediction_rf_model.joblib")
    scaler_path = os.path.join("models", "income_prediction_scaler.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(
            "⚠️  Model files not found. "
            "Please run the notebook first to generate "
            "`income_prediction_rf_model.joblib` and "
            "`income_prediction_scaler.joblib`, then place them in the `models/` folder."
        )
        st.stop()

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

# ── Known Training Columns ────────────────────────────────────────────────
# These are the exact columns produced by the notebook's one-hot encoding.
# They must match the columns the model was trained on.
TRAINING_COLUMNS = [
    'age', 'capital_gain', 'capital_loss', 'hours_per_week',
    'has_capital_gain', 'has_capital_loss',
    'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private',
    'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
    'workclass_State-gov', 'workclass_Without-pay',
    'education_11th', 'education_12th', 'education_1st-4th',
    'education_5th-6th', 'education_7th-8th', 'education_9th',
    'education_Assoc-acdm', 'education_Assoc-voc', 'education_Bachelors',
    'education_Doctorate', 'education_HS-grad', 'education_Masters',
    'education_Preschool', 'education_Prof-school', 'education_Some-college',
    'marital_status_Married-AF-spouse', 'marital_status_Married-civ-spouse',
    'marital_status_Married-spouse-absent', 'marital_status_Never-married',
    'marital_status_Separated', 'marital_status_Widowed',
    'occupation_Armed-Forces', 'occupation_Craft-repair',
    'occupation_Exec-managerial', 'occupation_Farming-fishing',
    'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
    'occupation_Other-service', 'occupation_Priv-house-serv',
    'occupation_Prof-specialty', 'occupation_Protective-serv',
    'occupation_Sales', 'occupation_Tech-support', 'occupation_Transport-moving',
    'relationship_Not-in-family', 'relationship_Other-relative',
    'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife',
    'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White',
    'sex_Male',
    'native_country_Cambodia', 'native_country_Canada', 'native_country_China',
    'native_country_Columbia', 'native_country_Cuba',
    'native_country_Dominican-Republic', 'native_country_Ecuador',
    'native_country_El-Salvador', 'native_country_England',
    'native_country_France', 'native_country_Germany', 'native_country_Greece',
    'native_country_Guatemala', 'native_country_Haiti', 'native_country_Honduras',
    'native_country_Hong', 'native_country_Hungary', 'native_country_India',
    'native_country_Iran', 'native_country_Ireland', 'native_country_Italy',
    'native_country_Jamaica', 'native_country_Japan', 'native_country_Laos',
    'native_country_Mexico', 'native_country_Nicaragua',
    'native_country_Outlying-US(Guam-USVI-etc)', 'native_country_Peru',
    'native_country_Philippines', 'native_country_Poland',
    'native_country_Portugal', 'native_country_Puerto-Rico',
    'native_country_Scotland', 'native_country_South', 'native_country_Taiwan',
    'native_country_Thailand', 'native_country_Trinadad&Tobago',
    'native_country_United-States', 'native_country_Vietnam',
    'native_country_Yugoslavia',
]

NUMERICAL_COLS = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']

# ── Preprocessing Function ────────────────────────────────────────────────
def preprocess_input(raw_input: dict) -> pd.DataFrame:
    """
    Convert raw user input dict into a model-ready DataFrame:
    1. Build engineered features
    2. One-hot encode categoricals
    3. Align columns to training schema (fill missing dummies with 0)
    4. Apply saved StandardScaler to numerical columns
    """
    df = pd.DataFrame([raw_input])

    # Engineer binary capital flags
    df['has_capital_gain'] = (df['capital_gain'] > 0).astype(int)
    df['has_capital_loss'] = (df['capital_loss'] > 0).astype(int)

    # One-hot encode (same logic as notebook: drop_first=True)
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'native_country']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Align to training columns — add any missing dummy columns as 0
    for col in TRAINING_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Keep only training columns, in the exact training order
    df = df[TRAINING_COLUMNS]

    # Scale numerical features using the saved scaler
    df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])

    return df


# ── App Header ────────────────────────────────────────────────────────────
st.title("💰 Income Level Predictor")
st.markdown(
    "This app predicts whether an individual earns **above or below \\$50,000/year** "
    "based on U.S. Census features, using a **Tuned Random Forest Classifier** trained on "
    "48,842 census records."
)
st.markdown("---")

# ── Sidebar — About ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About This App")
    st.markdown(
        """
        **Model:** Tuned Random Forest Classifier  
        **Dataset:** UCI Adult Census Income (1994)  
        **Features:** 14 (including 2 engineered)  
        **Primary metric:** ROC-AUC  

        **Top Predictors:**
        - Age
        - Capital Gain
        - Hours per Week
        - Marital Status
        - Education & Occupation

        ---
        **Author:** Adewale Adeagbo  
        [GitHub](https://github.com/cssadewale) · 
        [LinkedIn](https://linkedin.com/in/adewalesamsonadeagbo)
        """
    )

# ── Input Form ────────────────────────────────────────────────────────────
st.subheader("📋 Enter Individual Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=17, max_value=90, value=35, step=1)

    education = st.selectbox("Education Level", [
        "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th",
        "10th", "11th", "12th", "HS-grad", "Some-college",
        "Assoc-acdm", "Assoc-voc", "Bachelors",
        "Masters", "Prof-school", "Doctorate"
    ], index=12)

    occupation = st.selectbox("Occupation", [
        "Adm-clerical", "Armed-Forces", "Craft-repair",
        "Exec-managerial", "Farming-fishing", "Handlers-cleaners",
        "Machine-op-inspct", "Other-service", "Priv-house-serv",
        "Prof-specialty", "Protective-serv", "Sales",
        "Tech-support", "Transport-moving"
    ], index=3)

    marital_status = st.selectbox("Marital Status", [
        "Divorced", "Married-AF-spouse", "Married-civ-spouse",
        "Married-spouse-absent", "Never-married", "Separated", "Widowed"
    ], index=2)

    hours_per_week = st.slider(
        "Hours Worked per Week", min_value=1, max_value=99, value=40, step=1
    )

with col2:
    workclass = st.selectbox("Work Class", [
        "Federal-gov", "Local-gov", "Never-worked", "Private",
        "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"
    ], index=3)

    relationship = st.selectbox("Relationship Role", [
        "Husband", "Not-in-family", "Other-relative",
        "Own-child", "Unmarried", "Wife"
    ], index=0)

    sex = st.radio("Sex", ["Male", "Female"], index=0, horizontal=True)

    race = st.selectbox("Race", [
        "Amer-Indian-Eskimo", "Asian-Pac-Islander",
        "Black", "Other", "White"
    ], index=4)

    native_country = st.selectbox("Native Country", [
        "United-States", "Mexico", "Philippines", "Germany", "Canada",
        "Puerto-Rico", "El-Salvador", "India", "Cuba", "England",
        "Jamaica", "South", "China", "Italy", "Dominican-Republic",
        "Vietnam", "Guatemala", "Japan", "Poland", "Columbia",
        "Taiwan", "Haiti", "Iran", "Portugal", "Nicaragua",
        "Peru", "France", "Greece", "Ecuador", "Ireland",
        "Hong", "Cambodia", "Thailand", "Trinadad&Tobago",
        "Yugoslavia", "Outlying-US(Guam-USVI-etc)", "Hungary",
        "Honduras", "Scotland", "Laos", "Bangladesh"
    ], index=0)

# Capital gain and loss on full width
st.markdown("---")
st.markdown("**Investment Income (enter 0 if none)**")
cap_col1, cap_col2 = st.columns(2)
with cap_col1:
    capital_gain = st.number_input(
        "Capital Gain ($)", min_value=0, max_value=99999, value=0, step=100
    )
with cap_col2:
    capital_loss = st.number_input(
        "Capital Loss ($)", min_value=0, max_value=4356, value=0, step=100
    )

# ── Predict Button ────────────────────────────────────────────────────────
st.markdown("---")
predict_clicked = st.button("🔮 Predict Income Level", use_container_width=True)

if predict_clicked:
    # Assemble raw input
    raw_input = {
        "age":            age,
        "workclass":      workclass,
        "education":      education,
        "marital_status": marital_status,
        "occupation":     occupation,
        "relationship":   relationship,
        "race":           race,
        "sex":            sex,
        "capital_gain":   capital_gain,
        "capital_loss":   capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country,
    }

    # Preprocess
    X_input = preprocess_input(raw_input)

    # Predict
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]  # P(>50K)

    # ── Display Result ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.success(f"### ✅ Predicted Income: **> $50,000 / year**")
        st.markdown(
            f"The model estimates a **{probability:.1%} probability** "
            f"that this individual earns above \\$50,000 per year."
        )
    else:
        st.warning(f"### 📉 Predicted Income: **≤ $50,000 / year**")
        st.markdown(
            f"The model estimates a **{probability:.1%} probability** "
            f"that this individual earns above \\$50,000 per year."
        )

    # Probability progress bar
    st.markdown("**Probability of earning > $50K:**")
    st.progress(float(probability))
    st.caption(f"{probability:.1%} confidence")

    # ── Key Input Summary ─────────────────────────────────────────────────
    with st.expander("📄 View Input Summary"):
        summary = pd.DataFrame({
            "Feature": [
                "Age", "Education", "Occupation", "Marital Status",
                "Work Class", "Hours/Week", "Sex", "Race",
                "Native Country", "Capital Gain", "Capital Loss"
            ],
            "Value": [
                age, education, occupation, marital_status,
                workclass, hours_per_week, sex, race,
                native_country, f"${capital_gain:,}", f"${capital_loss:,}"
            ]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "⚠️  **Disclaimer:** This prediction is based on a model trained on 1994 U.S. Census data. "
        "It is intended for educational and portfolio demonstration purposes only. "
        "Real-world income is influenced by many factors not captured in this dataset. "
        "This tool should not be used to make decisions affecting real individuals without "
        "a formal fairness audit."
    )

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built by **Adewale Adeagbo** · "
    "[GitHub](https://github.com/cssadewale) · "
    "[LinkedIn](https://linkedin.com/in/adewalesamsonadeagbo) · "
    "YouThrive Data Science Capstone 2025"
)
