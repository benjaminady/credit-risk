"""Logistic regression baseline, and a demonstration of why two metrics are needed."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss

from load_data import load, split
from metrics import score, calibration_table


def fit_baseline(Xtr, ytr):
    # Scaling matters here: credit limits run to six figures while repayment
    # codes run from -2 to 8. Inside a pipeline so the scaler is fitted on the
    # training data only and never sees the test set.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)


if __name__ == "__main__":
    df = load()
    Xtr, Xte, ytr, yte = split(df)

    model = fit_baseline(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    s = score(yte, p)

    print(f"logistic regression   AUC {s['auc']:.4f}   Brier {s['brier']:.4f}")
    print(f"predict the base rate AUC 0.5000   Brier "
          f"{brier_score_loss(yte, np.full(len(yte), yte.mean())):.4f}")

    # Why Brier is reported at all. This model cheats by looking at the answer
    # and assigning 0.90 to every defaulter and 0.80 to everyone else. Its
    # ranking is perfect and its probabilities are worthless.
    cheat = np.where(yte.values == 1, 0.90, 0.80)
    c = score(yte, cheat)
    print(f"\na model that ranks perfectly and is useless:")
    print(f"                      AUC {c['auc']:.4f}   Brier {c['brier']:.4f}")

    print("\ncalibration by decile")
    print(calibration_table(yte, p).round(4).to_string())
