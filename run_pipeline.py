"""
run_pipeline.py — Orchestrator Script

Runs the full ML pipeline end-to-end:
  1. Generate synthetic data  (if not already present)
  2. Load & preprocess
  3. Train models  (with SMOTE)
  4. Evaluate & compare
  5. Save best model

Usage
-----
    python run_pipeline.py
"""

import os
import sys

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.generate_data import generate_fraud_dataset
from src.data_pipeline import load_and_prepare_data
from src.modeling import train_all_models, save_best_model
from src.evaluation import compare_models, print_comparison_table


def main():
    # ──── Step 1: Generate data (if needed) ─────────────────────────────
    data_path = os.path.join("data", "fraud_data.csv")
    if not os.path.exists(data_path):
        print("\n📁 Data file not found — generating synthetic dataset …")
        os.makedirs("data", exist_ok=True)
        df = generate_fraud_dataset(n_samples=10_000)
        df.to_csv(data_path, index=False)
        print(f"   Saved {len(df):,} rows → {data_path}\n")
    else:
        print(f"\n📁 Found existing data at {data_path}\n")

    # ──── Step 2: Data Pipeline ─────────────────────────────────────────
    print("━" * 60)
    print("STEP 2 ▶ DATA PIPELINE")
    print("━" * 60)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = (
        load_and_prepare_data(data_path)
    )

    # ──── Step 3: Model Training ────────────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 3 ▶ MODEL TRAINING  (with SMOTE)")
    print("━" * 60)
    trained_models, training_times = train_all_models(X_train, y_train)

    # ──── Step 4: Evaluation ────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 4 ▶ EVALUATION")
    print("━" * 60)
    comparison_df, all_metrics = compare_models(trained_models, X_test, y_test)
    print_comparison_table(comparison_df)

    # Print per-model classification reports
    for name, m in all_metrics.items():
        print(f"\n--- {name} ---")
        print(m["classification_report"])

    # ──── Step 5: Save Best Model ───────────────────────────────────────
    best_name = comparison_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    save_best_model(best_model, best_name)

    print("\n🎉 Pipeline complete! Launch the demo with:")
    print("   streamlit run app.py\n")


if __name__ == "__main__":
    main()
