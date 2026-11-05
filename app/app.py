import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# Load the v2 model, feature order, and threshold
# (all three files must sit in the same folder as this app.py)
# ============================================================
BASE = Path(__file__).parent


def find_file(*prefixes, suffix=".pkl"):
    """Find a saved file even if GitHub/Windows added '(2)' to its name."""
    for prefix in prefixes:
        exact = BASE / f"{prefix}{suffix}"
        if exact.exists():
            return exact
    for prefix in prefixes:
        matches = sorted(BASE.glob(f"{prefix}*{suffix}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No file found for {prefixes} in {BASE}")


@st.cache_resource
def load_artifacts():
    model = joblib.load(find_file("pmos_reduced_model", "pcos_reduced_model"))
    features = joblib.load(find_file("pmos_feature_order", "pcos_feature_order"))
    thr = json.load(open(find_file("pmos_threshold", suffix=".json")))["threshold"]
    return model, list(features), float(thr)


model, feature_order, THRESHOLD = load_artifacts()

# How "Cycle(R/I)" is coded in the training data. The public dataset codes
# regular = 2 and irregular = 4. Check with df["Cycle(R/I)"].value_counts()
# in the notebook and change these two numbers if your data differs.
CYCLE_REGULAR = 2
CYCLE_IRREGULAR = 4

st.set_page_config(page_title="PMOS Risk Estimator", page_icon="🩺", layout="centered")

st.title("PMOS (formerly PCOS) Risk Estimator")
st.caption("Research prototype — 10-feature Random Forest model (v2)")

st.warning(
    "⚠️ **This is a research prototype, not a diagnostic tool.** "
    "It was trained on a single public dataset of 541 patients and has not been clinically "
    "validated. Do not use it to make real medical decisions — please consult a "
    "qualified healthcare provider for diagnosis and treatment."
)

st.write(
    "Enter the values below. These are the 10 features ranked most important by SHAP "
    "feature-importance analysis in the underlying research project."
)

st.divider()

# ============================================================
# Input form
# ============================================================
with st.form("pmos_form"):
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

    submitted = st.form_submit_button("Estimate PMOS Likelihood", use_container_width=True)

# ============================================================
# Prediction
# ============================================================
if submitted:
    # Map form inputs to the exact feature names the model was trained on
    input_dict = {
        "Follicle No. (R)": follicle_r,
        "Follicle No. (L)": follicle_l,
        "Weight gain(Y/N)": 1 if weight_gain == "Yes" else 0,
        "Skin darkening (Y/N)": 1 if skin_darkening == "Yes" else 0,
        "hair growth(Y/N)": 1 if hair_growth == "Yes" else 0,
        "Cycle length(days)": cycle_length,
        "AMH(ng/mL)": amh,
        "Cycle(R/I)": CYCLE_IRREGULAR if cycle_regularity == "Irregular" else CYCLE_REGULAR,
        "Fast food (Y/N)": 1 if fast_food == "Yes" else 0,
        "Pimples(Y/N)": 1 if pimples == "Yes" else 0,
    }

    missing = [f for f in feature_order if f not in input_dict]
    if missing:
        st.error(f"The model expects features this form does not collect: {missing}")
        st.stop()

    # Build the input row in the exact order the model was trained on
    input_df = pd.DataFrame([[input_dict[f] for f in feature_order]], columns=feature_order)

    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = "PMOS Likely" if probability >= THRESHOLD else "PMOS Unlikely"

    st.divider()
    st.subheader("Result")

    if prediction == "PMOS Likely":
        st.error(f"**{prediction}** — model score: {probability:.0%}")
    else:
        st.success(f"**{prediction}** — model score: {probability:.0%}")

    st.progress(min(probability, 1.0))

    st.caption(
        f"Decision threshold: {THRESHOLD:.3f}. It was chosen on validation data to find at least "
        "90% of true cases, which means more false alarms (on average, recall about 0.90 and "
        "precision about 0.74 across 30 test splits)."
    )
    st.caption(
        "The score is a rough guide, not an exact percentage. In our calibration analysis, "
        "Random Forest probabilities were less reliable than those of Logistic Regression."
    )

    with st.expander("What do these numbers mean?"):
        st.write(
            "This result comes from a Random Forest model trained on 541 patients, using only "
            "the 10 features shown above out of 39 available. In our tests, no significant "
            "difference in performance was found between this 10-feature model and the full "
            "39-feature model (p = 0.97). The model relies strongly on follicle counts from "
            "ultrasound; when ultrasound features were removed, performance dropped clearly."
        )

st.divider()
st.caption(
    "Built as part of a research project on explainable machine learning for PMOS (formerly PCOS) "
    "screening. Model: Random Forest, mean ROC-AUC ≈ 0.95 over 30 random test splits "
    "(single final test: 0.91, 95% CI 0.79–0.99)."
)
