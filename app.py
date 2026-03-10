"""
Streamlit Demo — Fraud Detection System
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from src.data_pipeline import load_raw_data, clean_data, load_and_prepare_data
from src.modeling import train_all_models
from src.evaluation import plot_confusion_matrix, plot_roc_curves, plot_pr_curves, compare_models


# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fraud Detection System")


# -------------------------------------------------------
# DATA CHECK
# -------------------------------------------------------

DATA_PATH = os.path.join("data", "fraud_data.csv")

if not os.path.exists(DATA_PATH):
    st.warning("Dataset not found. Generating dataset and training model...")
    subprocess.run(["python", "run_pipeline.py"])
    st.success("Dataset generated successfully.")


# -------------------------------------------------------
# DATA LOADER
# -------------------------------------------------------

@st.cache_data
def load_data():
    df_raw = load_raw_data()
    df = clean_data(df_raw)
    return df


df = load_data()


# -------------------------------------------------------
# TRAINING PIPELINE
# -------------------------------------------------------

@st.cache_resource
def run_training_pipeline():

    X_train, X_test, y_train, y_test, preprocessor, feature_names = load_and_prepare_data()

    trained_models, training_times = train_all_models(X_train, y_train)

    comparison_df, all_metrics = compare_models(trained_models, X_test, y_test)

    return {
        "trained_models": trained_models,
        "comparison_df": comparison_df,
        "all_metrics": all_metrics,
        "training_times": training_times,
        "X_test": X_test,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names
    }


# -------------------------------------------------------
# TABS
# -------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    ["📊 Data Explorer", "🏆 Model Results", "🔮 Live Prediction"]
)

# =======================================================
# TAB 1 — DATA EXPLORER
# =======================================================

with tab1:

    st.header("Dataset Overview")

    n_fraud = int(df["is_fraud"].sum())
    n_legit = len(df) - n_fraud

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", len(df))
    col2.metric("Legitimate", n_legit)
    col3.metric("Fraud", n_fraud)
    col4.metric("Fraud Rate", f"{n_fraud/len(df):.2%}")

    st.divider()

    st.subheader("Class Distribution")

    fig, ax = plt.subplots()

    ax.bar(
        ["Legit", "Fraud"],
        [n_legit, n_fraud],
        color=["green", "red"]
    )

    st.pyplot(fig)

    st.subheader("Feature Statistics")

    st.dataframe(df.describe())


# =======================================================
# TAB 2 — MODEL RESULTS
# =======================================================

with tab2:

    st.header("Model Comparison")

    with st.spinner("Training models..."):

        results = run_training_pipeline()

    st.dataframe(results["comparison_df"])

    best_model = results["comparison_df"].iloc[0]["Model"]

    st.success(f"Best model: {best_model}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curve")
        fig = plot_roc_curves(results["all_metrics"], results["y_test"])
        st.pyplot(fig)

    with col2:
        st.subheader("PR Curve")
        fig = plot_pr_curves(results["all_metrics"], results["y_test"])
        st.pyplot(fig)


# =======================================================
# TAB 3 — LIVE PREDICTION
# =======================================================

with tab3:

    st.header("Live Fraud Prediction")

    with st.spinner("Loading models..."):

        results = run_training_pipeline()

        best_model_name = results["comparison_df"].iloc[0]["Model"]

        model = results["trained_models"][best_model_name]

        preprocessor = results["preprocessor"]

        feature_names = results["feature_names"]

    st.write("Enter transaction features")

    inputs = {}

    for feat in feature_names:
        inputs[feat] = st.number_input(feat, value=0.0)

    if st.button("Predict Fraud"):

        input_df = pd.DataFrame([inputs])

        X = preprocessor.transform(input_df)

        pred = model.predict(X)[0]

        proba = model.predict_proba(X)[0]

        if pred == 1:
            st.error("🚨 Fraud Detected")
        else:
            st.success("✅ Legitimate Transaction")

        st.metric("Fraud Probability", f"{proba[1]:.2%}")
