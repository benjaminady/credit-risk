# Credit Default Risk and the Approval Threshold

A default model on 30,000 real credit accounts, and an analysis of where the
decision threshold belongs once you account for the fact that keeping a bad
account and declining a good one cost different amounts.

The modelling is the easy half. The point of the project is the second half:
turning a probability into a decision, stating the assumptions that decision
rests on, and reporting how far the answer moves when those assumptions are wrong.

## Data

UCI *Default of Credit Card Clients* (Taiwan, 2005). 30,000 accounts, six months
of billing and repayment history, binary outcome of default in the following
month. Base default rate 22.1 percent. Stratified 75/25 train and test split.

## What decision is being modelled

Worth being precise, because it is easy to describe this wrongly. The accounts
already have six months of history, so this is not an origination decision. It is
a continued exposure decision: whether to keep extending credit on an existing
account or act on it by reducing the limit, tightening terms, or routing it to
collections. Exposure is therefore the most recent statement balance rather than
the credit limit.

## A deliberate constraint

The dataset includes sex, education, marital status and age. A US lender may not
use any of them in a credit decision under the Equal Credit Opportunity Act and
Regulation B. They are dropped before modelling. The model uses 19 behavioural
and account level features.

## Model results

| Model | AUC | Brier | Largest calibration gap |
|---|---|---|---|
| Predict the base rate | 0.500 | 0.1723 | n/a |
| Logistic regression | 0.7351 | 0.1427 | 0.132 (decile 10) |
| Gradient boosting | 0.7989 | 0.1299 | 0.040 (decile 9) |
| Gradient boosting, isotonic calibrated | 0.7989 | 0.1295 | 0.051 (decile 9) |

The isotonic row was tested and then dropped, so it is not in the shipped
scripts. It is reported because a recalibration that changed nothing is a result.

Brier is reported alongside AUC throughout because the threshold analysis depends
on the probabilities being approximately right, not merely well ordered.

**The calibration finding matters more than the AUC improvement.** Logistic
regression ranks acceptably but is badly wrong at the top of the risk
distribution: in the riskiest decile it predicts 57 percent and 71 percent
actually default. A lender using those probabilities to price or decline would
systematically underprice its worst accounts. Gradient boosting cuts that gap
from 0.132 to 0.040. Isotonic recalibration did not improve it further on this
split, which is worth saying rather than hiding: the uncalibrated boosted model
was already close enough that the recalibration added nothing.

Permutation importance, measured as the drop in AUC when a column is shuffled:

    PAY_0        0.089    repayment status last month
    LIMIT_BAL    0.021    credit limit
    BILL_AMT1    0.016    most recent statement balance
    PAY_AMT2     0.007
    PAY_2        0.006    repayment status two months ago

Whether someone paid last month dominates everything else by a factor of four.
That is not a surprising finding and it should not be dressed up as one. A model
that said otherwise would be a reason to check the model rather than to believe
the result.

Hyperparameters are left near defaults and the test set was not used for tuning.

## The threshold analysis

Keep an account and it does not default, you earn the margin on its balance.
Keep it and it defaults, you lose the loss given default on that balance. Act on
it and you earn nothing. So the rule is to keep accounts whose predicted
probability of default sits below some threshold.

Assumptions, which are assumptions rather than facts: margin of 8 percent of
exposure, loss given default of 75 percent of exposure. Both are
order of magnitude figures for unsecured revolving credit.

The economics alone imply a break even threshold of `margin / (margin + LGD)`,
which is 0.0964. The empirical optimum on the test set is 0.085, inside a plateau of 0.085 to
0.105 that holds 95 percent of the available profit. Theory landing inside that
band is the real validation of the calibration work:
if the predicted probabilities were badly wrong, the empirical optimum would not
land near the theoretical one.

| Policy | Threshold | Approval rate | Default rate among approved | Profit |
|---|---|---|---|---|
| Keep everyone | 1.01 | 100.0% | 22.1% | -38,536,561 |
| Naive 0.50 cutoff | 0.50 | 87.9% | 15.5% | -9,188,805 |
| Tighter 0.20 | 0.20 | 65.4% | 10.4% | -1,931,050 |
| Economic optimum | 0.085 | 19.9% | 4.3% | +4,579,393 |

A 0.50 cutoff is the default behaviour of almost every classifier and it loses
money heavily here. The gap between the naive cutoff and the economic optimum is
the entire value of doing this step, and it does not show up in any accuracy
metric.

One honest caveat on the numbers above: a portfolio with a 22 percent default
rate, a 75 percent loss given default and an 8 percent margin is unprofitable by
construction, which is why keeping everyone shows a large loss. This is a
distressed 2005 Taiwanese book rather than a normal consumer portfolio. The
absolute figures should not be read as a realistic profit and loss. The structure
of the decision, and the distance between the naive and optimal thresholds, are
what transfer.

## Sensitivity

Optimal threshold across a grid of assumptions:

|  | LGD 55% | LGD 65% | LGD 75% | LGD 85% | LGD 95% |
|---|---|---|---|---|---|
| margin 4% | 0.085 | 0.085 | 0.085 | 0.065 | 0.060 |
| margin 6% | 0.085 | 0.085 | 0.085 | 0.085 | 0.085 |
| margin 8% | 0.130 | 0.105 | 0.085 | 0.085 | 0.085 |
| margin 10% | 0.145 | 0.135 | 0.130 | 0.105 | 0.085 |
| margin 14% | 0.220 | 0.145 | 0.145 | 0.145 | 0.130 |

Holding margin fixed and varying loss given default moves the threshold by up to
0.090. Holding loss given default fixed and varying margin moves it by up to
0.135. **The margin assumption is roughly one and a half times as consequential
as the loss assumption** over these ranges, which is the opposite of where most
attention usually goes: credit teams spend their effort refining loss estimates
while the pricing assumption, often owned by a different team entirely, moves the
decision more. The ratio depends on the ranges tested, so it is quoted with them.

## Figures

    figures/calibration.png   predicted vs observed default rate by decile, all three models
    figures/threshold.png     expected profit across every threshold, optimum marked
    figures/sensitivity.png   optimal threshold across the margin and LGD grid

## Structure

    src/load_data.py          loading, protected attribute exclusion, stratified split
    src/metrics.py            shared AUC, Brier and decile calibration helpers
    src/step1_baseline.py     logistic regression baseline, and why AUC alone misleads
    src/step2_model.py        gradient boosting, calibration comparison, importance
    src/step3_threshold.py    expected profit, policy comparison, sensitivity grid
    DECISIONS.md              choices made and why

## Reproducing

    pip install pandas numpy scikit-learn matplotlib
    cd src
    python3 step1_baseline.py
    python3 step2_model.py        # writes data/test_predictions.csv
    python3 step3_threshold.py

Step 3 reads the saved predictions rather than refitting, so the modelling and
the economics stay separable.

## What I would do next

- Validate on a time based split rather than a random one. Default behaviour is
  not stationary and a random split is optimistic about how the model holds up.
- Test the model with the protected attributes included, privately, to quantify
  how much predictive power the ECOA exclusion actually costs. If the answer is
  close to nothing, that is worth knowing.
- Build a reject inference discussion. Acting on an account removes the outcome
  you would have observed, so a model retrained on surviving accounts drifts.

## Author

Benjamin Ady. M.S. Data Science, Eastern University, 2025.
