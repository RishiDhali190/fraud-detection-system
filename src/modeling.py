"""
modeling.py — Model Training with Imbalanced-Data Handling

Trains multiple classifiers on the fraud dataset and saves the best one.
Handles class imbalance via:
  • SMOTE (Synthetic Minority Over-sampling Technique)
  • class_weight='balanced' where supported

Models compared
---------------
  1. Logistic Regression
  2. Random Forest
  3. Gradient Boosting (sklearn)

Usage
-----
    from src.modeling import train_all_models
    results = train_all_models(X_train, y_train)
"""

import os
import time
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

MODELS_DIR = "models"


def apply_smote(X_train, y_train, random_state: int = 42):
    """Apply SMOTE to balance the training set.

    Returns
    -------
    X_resampled, y_resampled : np.ndarray
        Balanced arrays.
    """
    smote = SMOTE(random_state=random_state, sampling_strategy="auto")
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"🔁 SMOTE: {len(X_train):,} → {len(X_res):,} samples "
          f"(fraud went from {y_train.sum()} to {y_res.sum()})")
    return X_res, y_res


def get_models(random_state: int = 42) -> dict:
    """Return a dictionary of {name: estimator}."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
        ),
    }


def train_all_models(X_train, y_train, use_smote: bool = True, random_state: int = 42):
    """Train every model and return results.

    Parameters
    ----------
    X_train : np.ndarray   Preprocessed feature matrix.
    y_train : np.ndarray   Target labels.
    use_smote : bool       Whether to apply SMOTE before training.

    Returns
    -------
    trained_models : dict   {name: fitted_estimator}
    training_times : dict   {name: seconds}
    """
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state)

    models = get_models(random_state)
    trained_models = {}
    training_times = {}

    for name, model in models.items():
        print(f"🏋️ Training {name} …", end=" ")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        trained_models[name] = model
        training_times[name] = round(elapsed, 2)
        print(f"done in {elapsed:.2f}s")

    return trained_models, training_times


def save_best_model(model, model_name: str):
    """Persist the best model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(model, path)
    # Also save name for reference
    joblib.dump(model_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))
    print(f"💾 Best model saved: {model_name} → {path}")
