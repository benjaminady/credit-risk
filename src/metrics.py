"""Shared evaluation helpers, so every script measures things the same way."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss


def score(y_true, p) -> dict:
    """AUC measures ranking. Brier measures calibration. Both are needed:
    a model can rank perfectly and still produce unusable probabilities."""
    return {"auc": roc_auc_score(y_true, p), "brier": brier_score_loss(y_true, p)}


def calibration_table(y_true, p, n_bins: int = 10) -> pd.DataFrame:
    """Cut the predictions into equal sized groups and compare what the model
    said would happen against what did.

    Equal sized rather than equal width, because predictions bunch near the base
    rate and equal width bins would leave the high risk ones nearly empty.
    """
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d["p"], q=n_bins, labels=False, duplicates="drop")
    t = (d.groupby("decile")
           .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")))
    t["gap"] = t["observed"] - t["predicted"]      # positive = risk understated
    return t
