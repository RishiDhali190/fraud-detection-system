"""
evaluation.py — Evaluation Beyond Accuracy

Computes metrics that matter for highly imbalanced fraud detection:
  • Precision, Recall, F1-Score  (for the fraud class)
  • ROC-AUC
  • Precision-Recall AUC  (PR-AUC, most informative for rare events)
  • Confusion Matrix
  • Full classification report

Also includes plotting helpers used by the Streamlit app.

Usage
-----
    from src.evaluation import evaluate_model, compare_models
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Compute all relevant metrics for a single model.

    Returns a dict with scalar metrics + arrays for curve plotting.
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Legit", "Fraud"]
        ),
        # For curve plotting
        "y_prob": y_prob,
        "y_pred": y_pred,
    }
    return metrics


def compare_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate all models and return a comparison DataFrame sorted by PR-AUC."""
    results = []
    all_metrics = {}
    for name, model in trained_models.items():
        m = evaluate_model(model, X_test, y_test, model_name=name)
        all_metrics[name] = m
        results.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
        })

    df = pd.DataFrame(results).sort_values("PR-AUC", ascending=False)
    return df, all_metrics


# ─────────────────────── plotting helpers ────────────────────────────


def plot_confusion_matrix(cm, title: str = "Confusion Matrix"):
    """Return a matplotlib Figure for a confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc_curves(all_metrics: dict, y_test):
    """Return a Figure with overlaid ROC curves for every model."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, m in all_metrics.items():
        fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
        ax.plot(fpr, tpr, label=f'{name} (AUC={m["roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr_curves(all_metrics: dict, y_test):
    """Return a Figure with overlaid Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, m in all_metrics.items():
        prec, rec, _ = precision_recall_curve(y_test, m["y_prob"])
        ax.plot(rec, prec, label=f'{name} (AP={m["pr_auc"]:.3f})')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def print_comparison_table(comparison_df: pd.DataFrame):
    """Pretty-print the comparison table."""
    print("\n" + "=" * 80)
    print("📊  MODEL COMPARISON  (sorted by PR-AUC)")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80 + "\n")
