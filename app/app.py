import streamlit as st
import pandas as pd
import joblib

# ============================================================
# Load the trained model and expected feature order
# (these .pkl files must sit in the same folder as this app.py)
# ============================================================
model = joblib.load("pcos_reduced_model.pkl")
feature_order = joblib.load("pcos_feature_order.pkl")

# Classification threshold chosen during development to prioritize recall
# (catches ~92% of true PCOS cases at ~80% precision - see project notebook)
THRESHOLD = 0.345

st.set_page_config(page_title="PCOS Risk Estimator", page_icon="🩺", layout="centered")

st.title("PCOS Risk Estimator")
st.caption("Research prototype — 10-feature Random Forest model")

st.warning(
    "⚠️ **This is a research prototype, not a diagnostic tool.** "
    "It was trained on a single dataset of 541 patients and has not been clinically "
    "validated. Do not use this to make real medical decisions — please consult a "
    "qualified healthcare provider for PCOS diagnosis and treatment."
)

st.write(
    "Enter the values below. These are the 10 features identified as most predictive "
    "of PCOS by SHAP feature-importance analysis in the underlying research project."
)

st.divider()

# ============================================================
# Input form
# ============================================================
with st.form("pcos_form"):
    col1, col2 = st.columns(2)

    with col1:
        follicle_r = st.number_input(
            "Follicle count — right ovary", min_value=0, max_value=40, value=8,
            help="Number of follicles visible on ultrasound, right ovary."
        )
        follicle_l = st.number_input(
            "Follicle count — left ovary", min_value=0, max_value=40, value=8,
            help="Number of follicles visible on ultrasound, left ovary."
        )
        weight_gain = st.radio("Recent unexplained weight gain?", ["No", "Yes"], horizontal=True)
        skin_darkening = st.radio("Skin darkening (e.g. neck, underarms)?", ["No", "Yes"], horizontal=True)
        hair_growth = st.radio("Excess facial/body hair growth?", ["No", "Yes"], horizontal=True)

    with col2:
        cycle_length = st.number_input(
            "Average menstrual cycle length (days)", min_value=10, max_value=90, value=28
        )
        amh = st.number_input(
            "AMH level (ng/mL)", min_value=0.0, max_value=30.0, value=3.0, step=0.1,
            help="Anti-Müllerian Hormone — a blood test result."
        )
        cycle_regularity = st.radio("Menstrual cycle regularity", ["Regular", "Irregular"], horizontal=True)
        fast_food = st.radio("Frequent fast food consumption?", ["No", "Yes"], horizontal=True)
        pimples = st.radio("Frequent acne/pimples?", ["No", "Yes"], horizontal=True)

    submitted = st.form_submit_button("Estimate PCOS Likelihood", use_container_width=True)

# ============================================================
# Prediction
# ============================================================
if submitted:
    # Map form inputs to the exact feature names/order the model expects
    input_dict = {
        "Follicle No. (R)": follicle_r,
        "Follicle No. (L)": follicle_l,
        "Weight gain(Y/N)": 1 if weight_gain == "Yes" else 0,
        "Skin darkening (Y/N)": 1 if skin_darkening == "Yes" else 0,
        "hair growth(Y/N)": 1 if hair_growth == "Yes" else 0,
        "Cycle length(days)": cycle_length,
        "AMH(ng/mL)": amh,
        "Cycle(R/I)": 1 if cycle_regularity == "Irregular" else 0,
        "Fast food (Y/N)": 1 if fast_food == "Yes" else 0,
        "Pimples(Y/N)": 1 if pimples == "Yes" else 0,
    }

    # Build the input row in the exact order the model was trained on
    input_df = pd.DataFrame([[input_dict[feat] for feat in feature_order]], columns=feature_order)

    probability = model.predict_proba(input_df)[0, 1]
    prediction = "PCOS Likely" if probability >= THRESHOLD else "PCOS Unlikely"

    st.divider()
    st.subheader("Result")

    if prediction == "PCOS Likely":
        st.error(f"**{prediction}** — estimated probability: {probability:.1%}")
    else:
        st.success(f"**{prediction}** — estimated probability: {probability:.1%}")

    st.progress(min(float(probability), 1.0))

    st.caption(
        f"Classification threshold used: {THRESHOLD:.1%} "
        "(optimized during development to prioritize catching true cases, "
        "at some cost to false-positive rate)."
    )

    with st.expander("What do these numbers mean?"):
        st.write(
            "This estimate comes from a Random Forest model trained on 541 patients, "
            "using only the 10 clinical features shown above out of 40+ originally available "
            "— chosen because they retained equivalent predictive performance "
            "(statistically indistinguishable, p = 0.89) to the full feature set, "
            "making this a more practical, lower-cost screening approach."
        )

st.divider()
st.caption(
    "Built as part of a research project on explainable machine learning for PCOS "
    "screening. Model: Random Forest, ROC-AUC ≈ 0.94 on held-out test data."
)
