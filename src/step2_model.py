"""Gradient boosting, and the calibration comparison that justifies choosing it."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from load_data import load, split
from metrics import score, calibration_table
from step1_baseline import fit_baseline

FIG = Path(__file__).resolve().parents[1] / "figures"
DATA = Path(__file__).resolve().parents[1] / "data"


def fit_model(Xtr, ytr):
    # No scaling: trees split on values, so units are irrelevant.
    # early_stopping holds back 15 percent of the training data and stops adding
    # trees when that stops improving, so the number of iterations is not a
    # hyperparameter to tune against the test set.
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=0.15, random_state=7,
    ).fit(Xtr, ytr)


def plot_calibration(tables: dict, path):
    fig, ax = plt.subplots(figsize=(6.2, 6))
    lim = max(max(t["predicted"].max(), t["observed"].max()) for t in tables.values())
    lim = min(1.0, lim * 1.1)
    for label, t in tables.items():
        ax.plot(t["predicted"], t["observed"], marker="o", lw=1.6, label=label)
    ax.plot([0, lim], [0, lim], ls="--", c="#888", lw=1, label="perfect calibration")
    ax.set_xlabel("mean predicted probability of default")
    ax.set_ylabel("observed default rate")
    ax.set_title("Calibration by decile of predicted risk")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(frameon=False, loc="upper left"); ax.grid(alpha=.25, lw=.5)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    df = load()
    Xtr, Xte, ytr, yte = split(df)

    p_lr = fit_baseline(Xtr, ytr).predict_proba(Xte)[:, 1]
    model = fit_model(Xtr, ytr)
    p_gb = model.predict_proba(Xte)[:, 1]

    for name, p in [("logistic regression", p_lr), ("gradient boosting", p_gb)]:
        s = score(yte, p)
        print(f"{name:<22} AUC {s['auc']:.4f}   Brier {s['brier']:.4f}")

    tables = {"logistic regression": calibration_table(yte, p_lr),
              "gradient boosting": calibration_table(yte, p_gb)}

    print("\ngradient boosting, calibration by decile")
    print(tables["gradient boosting"].round(4).to_string())

    for name, t in tables.items():
        mono = t["observed"].is_monotonic_increasing
        print(f"\n{name}: worst gap {t['gap'].abs().max():.4f}, "
              f"observed rate monotonic: {mono}")

    plot_calibration(tables, FIG / "calibration.png")

    # Permutation importance: shuffle a column, measure how much AUC falls.
    # More honest than tree-based importance, which inflates high-cardinality
    # features regardless of whether the model relies on them.
    imp = permutation_importance(model, Xte, yte, n_repeats=5,
                                 random_state=7, scoring="roc_auc", n_jobs=-1)
    print("\npermutation importance, top 6 (drop in AUC when shuffled)")
    for i in np.argsort(imp.importances_mean)[::-1][:6]:
        print(f"  {Xte.columns[i]:<12} {imp.importances_mean[i]:.4f}")

    # Hand the predictions to step 3 so the economics step refits nothing.
    pd.DataFrame({
        "y": yte.values,
        "p": p_gb,
        "exposure": Xte["BILL_AMT1"].clip(lower=0).values,
    }).to_csv(DATA / "test_predictions.csv", index=False)
    print(f"\nwrote {DATA / 'test_predictions.csv'}")
