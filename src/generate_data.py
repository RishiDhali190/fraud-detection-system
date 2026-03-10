"""
generate_data.py — Synthetic Fraud Dataset Generator

Generates a realistic, highly imbalanced credit-card-style dataset and saves
it to  data/fraud_data.csv .  The fraud class makes up ~1.7 % of samples,
mimicking real-world skew.

Usage
-----
    python -m src.generate_data          # from project root
    python src/generate_data.py          # also works
"""

import os
import numpy as np
import pandas as pd


def generate_fraud_dataset(
    n_samples: int = 10_000,
    fraud_ratio: float = 0.017,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a synthetic credit-card fraud dataset.

    Features
    --------
    - V1 … V20  : PCA-like continuous features (float64)
    - Amount     : transaction amount  (positive float)
    - Hour       : hour of day the transaction occurred (0-23)
    - Class      : 0 = legitimate, 1 = fraud  (target)

    Parameters
    ----------
    n_samples : int
        Total number of transactions to generate.
    fraud_ratio : float
        Proportion of fraudulent transactions (0 < ratio < 1).
    random_state : int
        Seed for reproducibility.
    """
    rng = np.random.RandomState(random_state)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # ---- Legitimate transactions ----------------------------------------
    legit_features = rng.randn(n_legit, 20)  # standard normal
    legit_amount = rng.exponential(scale=80, size=n_legit)
    legit_hour = rng.choice(range(24), size=n_legit, p=_hour_weights("legit"))

    # ---- Fraudulent transactions ----------------------------------------
    fraud_features = rng.randn(n_fraud, 20) * 1.5 + 0.8  # shifted & wider
    fraud_amount = rng.exponential(scale=250, size=n_fraud)
    fraud_hour = rng.choice(range(24), size=n_fraud, p=_hour_weights("fraud"))

    # ---- Combine --------------------------------------------------------
    features = np.vstack([legit_features, fraud_features])
    amounts = np.concatenate([legit_amount, fraud_amount])
    hours = np.concatenate([legit_hour, fraud_hour])
    labels = np.array([0] * n_legit + [1] * n_fraud)

    col_names = [f"V{i}" for i in range(1, 21)]
    df = pd.DataFrame(features, columns=col_names)
    df["Amount"] = np.round(amounts, 2)
    df["Hour"] = hours.astype(int)
    df["Class"] = labels

    # Inject ~2 % missing values at random positions (realistic messiness)
    mask = rng.rand(*df.iloc[:, :20].shape) < 0.02
    df.iloc[:, :20] = df.iloc[:, :20].mask(mask)

    # Shuffle rows
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def _hour_weights(kind: str) -> list:
    """Return a probability distribution over 24 hours."""
    if kind == "fraud":
        # Fraudsters prefer late-night / early-morning
        w = [5, 5, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5]
    else:
        # Legitimate transactions peak during daytime
        w = [1, 1, 1, 1, 1, 2, 3, 5, 6, 7, 7, 7,
             7, 7, 7, 6, 6, 5, 4, 3, 2, 2, 1, 1]
    total = sum(w)
    return [x / total for x in w]


def main():
    """Generate dataset and save to disk."""
    print("🔄 Generating synthetic fraud dataset …")
    df = generate_fraud_dataset(n_samples=10_000)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "fraud_data.csv")
    df.to_csv(out_path, index=False)

    fraud_count = df["Class"].sum()
    total = len(df)
    print(f"✅ Saved {total:,} rows to {out_path}")
    print(f"   Fraud: {fraud_count} ({fraud_count/total:.2%})  |  "
          f"Legit: {total - fraud_count}")


if __name__ == "__main__":
    main()
