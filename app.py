"""
app.py — Streamlit Demo for Fraud Detection System

Provides an interactive dashboard with three tabs:
  1. 📊 Data Explorer   — dataset overview, class distribution, feature stats
  2. 🏆 Model Results   — comparison table, ROC & PR curves, confusion matrices
  3. 🔮 Live Prediction — enter transaction features and get a fraud probability

Launch
------
    streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project imports work
sys.path.insert(0, os.path.dirname(__file__))

from src.data_pipeline import load_raw_data, clean_data
from src.evaluation import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    compare_models,
)
from src.modeling import train_all_models, get_models
from src.data_pipeline import load_and_prepare_data

# ────────────────────── page config ──────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────── custom CSS ───────────────────────────────────

st.markdown("""
<style>
    .stMetric {border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px;}
    .block-container {padding-top: 2rem;}
    h1 {color: #1a1a2e;}
</style>
""", unsafe_allow_html=True)

# ────────────────────── sidebar ──────────────────────────────────────

st.sidebar.title("🛡️ Fraud Detection")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**End-to-End ML System**\n\n"
    "A production-style fraud detection pipeline demonstrating:\n"
    "- Data cleaning\n"
    "- Imbalanced-data handling (SMOTE)\n"
    "- Model comparison\n"
    "- Evaluation beyond accuracy\n"
)

# ────────────────────── cached loaders ───────────────────────────────


@st.cache_data
def load_data():
    """Load and return raw & cleaned DataFrames."""
    df_raw = load_raw_data()
    df = clean_data(df_raw)
    return df


@st.cache_resource
def run_training_pipeline():
    """Run the ML pipeline and cache results."""
    X_train, X_test, y_train, y_test, preprocessor, feature_names = (
        load_and_prepare_data()
    )
    trained_models, training_times = train_all_models(X_train, y_train)
    comparison_df, all_metrics = compare_models(trained_models, X_test, y_test)
    return {
        "X_test": X_test,
        "y_test": y_test,
        "trained_models": trained_models,
        "training_times": training_times,
        "comparison_df": comparison_df,
        "all_metrics": all_metrics,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }


# ────────────────────── data check ───────────────────────────────────

DATA_PATH = os.path.join("data", "fraud_data.csv")
if not os.path.exists(DATA_PATH):
    st.error(
        "⚠️ Data file not found. Please run `python run_pipeline.py` first "
        "to generate the dataset and train models."
    )
    st.stop()

df = load_data()

# ────────────────────── main tabs ────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "🏆 Model Results", "🔮 Live Prediction"])

# ═══════════════════════════════════════════════════════════════════════
#  TAB 1 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════

with tab1:
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    n_fraud = int(df["Class"].sum())
    n_legit = len(df) - n_fraud
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Legitimate", f"{n_legit:,}")
    col3.metric("Fraudulent", f"{n_fraud:,}")
    col4.metric("Fraud Rate", f"{n_fraud/len(df):.2%}")

    st.markdown("---")

    # Class distribution
    left, right = st.columns(2)
    with left:
        st.subheader("Class Distribution")
        fig_dist, ax_dist = plt.subplots(figsize=(5, 3.5))
        colors = ["#2ecc71", "#e74c3c"]
        bars = ax_dist.bar(["Legitimate", "Fraud"], [n_legit, n_fraud], color=colors)
        for bar, val in zip(bars, [n_legit, n_fraud]):
            ax_dist.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                         f"{val:,}", ha="center", fontsize=11, fontweight="bold")
        ax_dist.set_ylabel("Count")
        ax_dist.set_title("Transactions by Class")
        fig_dist.tight_layout()
        st.pyplot(fig_dist)

    with right:
        st.subheader("Transaction Amount Distribution")
        fig_amt, ax_amt = plt.subplots(figsize=(5, 3.5))
        df[df["Class"] == 0]["Amount"].hist(
            bins=50, ax=ax_amt, alpha=0.7, label="Legit", color="#2ecc71"
        )
        df[df["Class"] == 1]["Amount"].hist(
            bins=50, ax=ax_amt, alpha=0.7, label="Fraud", color="#e74c3c"
        )
        ax_amt.set_xlabel("Amount ($)")
        ax_amt.set_ylabel("Frequency")
        ax_amt.set_title("Amount Distribution by Class")
        ax_amt.legend()
        fig_amt.tight_layout()
        st.pyplot(fig_amt)

    st.markdown("---")

    # Feature statistics
    st.subheader("Feature Statistics")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.subheader("Missing Values")
        st.bar_chart(missing)
    else:
        st.success("✅ No missing values after cleaning.")

    # Hour distribution
    st.subheader("Transactions by Hour of Day")
    fig_hour, ax_hour = plt.subplots(figsize=(8, 3.5))
    legit_hours = df[df["Class"] == 0]["Hour"].value_counts().sort_index()
    fraud_hours = df[df["Class"] == 1]["Hour"].value_counts().sort_index()
    ax_hour.bar(legit_hours.index - 0.2, legit_hours.values, width=0.4,
                label="Legit", color="#2ecc71", alpha=0.8)
    ax_hour.bar(fraud_hours.index + 0.2, fraud_hours.values, width=0.4,
                label="Fraud", color="#e74c3c", alpha=0.8)
    ax_hour.set_xlabel("Hour of Day")
    ax_hour.set_ylabel("Count")
    ax_hour.legend()
    fig_hour.tight_layout()
    st.pyplot(fig_hour)

# ═══════════════════════════════════════════════════════════════════════
#  TAB 2 — MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════

with tab2:
    st.header("🏆 Model Comparison & Evaluation")

    with st.spinner("Training models (cached after first run) …"):
        res = run_training_pipeline()

    # --- Comparison table ---
    st.subheader("Metrics Comparison")
    styled_df = (
        res["comparison_df"]
        .style
        .highlight_max(subset=["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
                       color="#d4edda")
        .format("{:.4f}", subset=["Accuracy", "Precision", "Recall", "F1-Score",
                                   "ROC-AUC", "PR-AUC"])
    )
    st.dataframe(styled_df, use_container_width=True)

    best_name = res["comparison_df"].iloc[0]["Model"]
    st.success(f"🥇 **Best model (by PR-AUC): {best_name}**")

    st.markdown("---")

    # --- Curves ---
    col_roc, col_pr = st.columns(2)
    with col_roc:
        st.subheader("ROC Curves")
        fig_roc = plot_roc_curves(res["all_metrics"], res["y_test"])
        st.pyplot(fig_roc)

    with col_pr:
        st.subheader("Precision-Recall Curves")
        fig_pr = plot_pr_curves(res["all_metrics"], res["y_test"])
        st.pyplot(fig_pr)

    st.markdown("---")

    # --- Confusion matrices ---
    st.subheader("Confusion Matrices")
    cm_cols = st.columns(len(res["all_metrics"]))
    for col, (name, m) in zip(cm_cols, res["all_metrics"].items()):
        with col:
            fig_cm = plot_confusion_matrix(m["confusion_matrix"], title=name)
            st.pyplot(fig_cm)

    # --- Training times ---
    st.markdown("---")
    st.subheader("Training Times")
    time_data = pd.DataFrame(
        list(res["training_times"].items()), columns=["Model", "Seconds"]
    )
    st.bar_chart(time_data.set_index("Model"))

# ═══════════════════════════════════════════════════════════════════════
#  TAB 3 — LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════

with tab3:
    st.header("🔮 Real-Time Fraud Prediction")
    st.markdown("Enter transaction details below or load a random sample from the test set.")

    # Use the cached training pipeline (avoids pickle deserialization issues)
    with st.spinner("Loading models (cached after first run) …"):
        res = run_training_pipeline()
        best_name = res["comparison_df"].iloc[0]["Model"]
        best_model = res["trained_models"][best_name]
        preprocessor = res["preprocessor"]
        feature_names = res["feature_names"]

    # Random sample loader
    if st.button("🎲 Load Random Sample"):
        sample = df.sample(1).iloc[0]
        for feat in feature_names:
            st.session_state[f"feat_{feat}"] = float(0.0 if pd.isna(sample[feat]) else sample[feat])
        if sample["Class"] == 1:
            st.info("ℹ️ This sample is **actually fraudulent** in the dataset.")
        else:
            st.info("ℹ️ This sample is **actually legitimate** in the dataset.")

    # Input form
    with st.form("prediction_form"):
        cols = st.columns(4)
        inputs = {}
        for i, feat in enumerate(feature_names):
            default = st.session_state.get(f"feat_{feat}", 0.0)
            with cols[i % 4]:
                inputs[feat] = st.number_input(
                    feat, value=default, format="%.4f", key=f"input_{feat}"
                )

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([inputs])
        X_input = preprocessor.transform(input_df)
        prediction = best_model.predict(X_input)[0]
        proba = best_model.predict_proba(X_input)[0]

        st.markdown("---")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if prediction == 1:
                st.error("🚨 **FRAUD DETECTED**")
            else:
                st.success("✅ **Transaction appears LEGITIMATE**")
        with col_res2:
            st.metric("Fraud Probability", f"{proba[1]:.2%}")
            st.metric("Legit Probability", f"{proba[0]:.2%}")

        # Probability bar
        fig_prob, ax_prob = plt.subplots(figsize=(6, 1.2))
        ax_prob.barh([""], [proba[0]], color="#2ecc71", label="Legit")
        ax_prob.barh([""], [proba[1]], left=[proba[0]], color="#e74c3c", label="Fraud")
        ax_prob.set_xlim(0, 1)
        ax_prob.legend(loc="upper right", fontsize=8)
        ax_prob.set_title("Prediction Confidence")
        fig_prob.tight_layout()
        st.pyplot(fig_prob)
