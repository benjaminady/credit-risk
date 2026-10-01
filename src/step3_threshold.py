"""Turning the probabilities into a decision.

Reads the saved test set predictions so nothing is refitted here. The modelling
and the economics are separate problems and keeping them in separate scripts
makes it obvious which assumptions belong to which.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
PRED = ROOT / "data" / "test_predictions.csv"

# Assumptions, stated as assumptions. Order of magnitude figures for unsecured
# revolving credit, not facts about this portfolio.
MARGIN = 0.08      # earned on the balance of an account that pays
LGD = 0.75         # lost on the balance of an account that defaults

# Every threshold on a 0.005 grid. Fine enough to locate the optimum, coarse
# enough that the grid itself is not pretending to a precision the data lacks.
GRID = np.arange(0.005, 1.000, 0.005)


def profit(y, p, exposure, threshold, margin=MARGIN, lgd=LGD):
    """Keep the account if predicted default probability is below the threshold.

    Kept and it pays      -> earn margin on the balance
    Kept and it defaults  -> lose lgd on the balance
    Acted on              -> earn nothing, lose nothing
    """
    keep = p < threshold
    earned = margin * exposure[keep & (y == 0)].sum()
    lost = lgd * exposure[keep & (y == 1)].sum()
    return earned - lost


def sweep(y, p, exposure, margin=MARGIN, lgd=LGD, grid=GRID) -> pd.DataFrame:
    rows = []
    for t in grid:
        keep = p < t
        n = keep.sum()
        rows.append({
            "threshold": t,
            "approval_rate": n / len(y),
            "default_rate_approved": y[keep].mean() if n else np.nan,
            "profit": profit(y, p, exposure, t, margin, lgd),
        })
    return pd.DataFrame(rows)


def plot_sweep(s, band, breakeven, path):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(s["threshold"], s["profit"] / 1e6, lw=1.8, color="#1F3A5F")
    ax.axhline(0, color="#888", lw=1, ls="--")
    ax.axvspan(band[0], band[1], color="#1F3A5F", alpha=.12,
               label=f"within 5 percent of peak: {band[0]:.3f} to {band[1]:.3f}")
    ax.axvline(MARGIN / (MARGIN + LGD), color="#B3541E", lw=1.3,
               label=f"theoretical break even: {MARGIN / (MARGIN + LGD):.4f}")
    if breakeven is not None:
        ax.axvline(breakeven, color="#555", lw=1.1, ls=":",
                   label=f"profit turns negative above: {breakeven:.3f}")
    ax.set_xlabel("threshold on predicted probability of default")
    ax.set_ylabel("expected profit, millions")
    ax.set_title("Expected profit across every approval threshold")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(alpha=.25, lw=.5)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_sensitivity(grid_df, path):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    im = ax.imshow(grid_df.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(grid_df.columns)))
    ax.set_xticklabels([f"{c:.0%}" for c in grid_df.columns])
    ax.set_yticks(range(len(grid_df.index)))
    ax.set_yticklabels([f"{r:.0%}" for r in grid_df.index])
    ax.set_xlabel("loss given default")
    ax.set_ylabel("margin")
    ax.set_title("Optimal threshold across the assumption grid")
    for i in range(grid_df.shape[0]):
        for j in range(grid_df.shape[1]):
            ax.text(j, i, f"{grid_df.values[i, j]:.3f}", ha="center",
                    va="center", color="w", fontsize=8.5)
    fig.colorbar(im, ax=ax, label="optimal threshold")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    d = pd.read_csv(PRED)
    y, p, exposure = d["y"].values, d["p"].values, d["exposure"].values
    print(f"accounts {len(y):,}   exposure at risk {exposure.sum():,.0f}")

    s = sweep(y, p, exposure)
    best = s.loc[s["profit"].idxmax()]
    theoretical = MARGIN / (MARGIN + LGD)
    print(f"\nempirical optimum  {best['threshold']:.3f}")
    print(f"theoretical        {theoretical:.4f}")

    # Policy comparison. Keeping everyone and the default 0.50 cutoff are the two
    # things that happen when nobody does this step.
    print("\npolicy comparison")
    print(f"{'policy':<22}{'thresh':>8}{'approved':>10}{'bad rate':>10}{'profit':>16}")
    policies = [("keep everyone", 1.01), ("naive 0.50 cutoff", 0.50),
                ("tighter 0.20", 0.20), ("economic optimum", best["threshold"]),
                ("over tight 0.05", 0.05)]
    for name, t in policies:
        keep = p < t
        pr = profit(y, p, exposure, t)
        print(f"{name:<22}{t:>8.3f}{keep.mean():>9.1%}"
              f"{y[keep].mean():>10.1%}{pr:>16,.0f}")

    # The plateau. On 7,500 accounts the exact peak is sampling noise, so the
    # reportable answer is the range that captures nearly all the profit.
    near = s[s["profit"] >= 0.95 * best["profit"]]["threshold"]
    band = (near.min(), near.max())
    # The last threshold still in profit, not the first one in loss, so the
    # number quoted is one you could actually operate at.
    loose = s[(s["threshold"] > best["threshold"]) & (s["profit"] < 0)]
    breakeven = (s.loc[loose.index[0] - 1, "threshold"]
                 if len(loose) else None)
    print(f"\nwithin 5% of peak profit: {band[0]:.3f} to {band[1]:.3f}")
    if breakeven is not None:
        print(f"profit turns negative above: {breakeven:.3f}")

    plot_sweep(s, band, breakeven, FIG / "threshold.png")

    # Sensitivity. The threshold is a function of two numbers nobody measured
    # exactly, so the honest deliverable includes how much they matter.
    margins = [0.04, 0.06, 0.08, 0.10, 0.14]
    lgds = [0.55, 0.65, 0.75, 0.85, 0.95]
    grid_df = pd.DataFrame(index=margins, columns=lgds, dtype=float)
    for m in margins:
        for L in lgds:
            ss = sweep(y, p, exposure, margin=m, lgd=L)
            grid_df.loc[m, L] = ss.loc[ss["profit"].idxmax(), "threshold"]

    print("\noptimal threshold by assumption (rows margin, columns LGD)")
    print(grid_df.round(3).to_string())

    by_lgd = (grid_df.max(axis=0) - grid_df.min(axis=0)).max()   # vary margin
    by_margin = (grid_df.max(axis=1) - grid_df.min(axis=1)).max()  # vary LGD
    print(f"\nvarying LGD moves it by    {by_margin:.3f}")
    print(f"varying margin moves it by {by_lgd:.3f}")

    plot_sensitivity(grid_df, FIG / "sensitivity.png")
    print(f"\nwrote {FIG / 'threshold.png'} and {FIG / 'sensitivity.png'}")
