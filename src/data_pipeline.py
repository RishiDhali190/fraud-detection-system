"""
data_pipeline.py — Data Loading, Cleaning & Preprocessing

Provides a clean, reproducible pipeline that:
  1. Loads raw CSV data
  2. Handles missing values (median imputation)
  3. Scales numerical features (StandardScaler)
  4. Splits into train / test sets (stratified)

Usage
-----
    from src.data_pipeline import load_and_prepare_data
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare_data()
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


DATA_PATH = os.path.join("data", "fraud_data.csv")
ARTIFACTS_DIR = "models"


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and perform basic validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Run `python -m src.generate_data` first."
        )
    df = pd.read_csv(path)
    required = {"Class", "Amount", "Hour"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop exact duplicates, enforce types."""
    df = df.drop_duplicates()
    df["Class"] = df["Class"].astype(int)
    df["Hour"] = df["Hour"].astype(int)
    return df


def build_preprocessor() -> Pipeline:
    """Create an sklearn Pipeline for imputation + scaling."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def load_and_prepare_data(
    path: str = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """End-to-end: load → clean → split → fit preprocessor → transform.

    Returns
    -------
    X_train, X_test : np.ndarray
        Scaled feature matrices.
    y_train, y_test : np.ndarray
        Target arrays.
    preprocessor : sklearn Pipeline
        Fitted imputer + scaler (saved for inference).
    feature_names : list[str]
        Original feature column names.
    """
    df = load_raw_data(path)
    df = clean_data(df)

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols]
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Persist preprocessor for the Streamlit app
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"))
    joblib.dump(feature_cols, os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))

    print(f"📊 Data loaded: {len(df):,} rows, {len(feature_cols)} features")
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"   Fraud rate (train): {y_train.mean():.2%}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_cols


if __name__ == "__main__":
    load_and_prepare_data()
    print("✅ Data pipeline complete.")
