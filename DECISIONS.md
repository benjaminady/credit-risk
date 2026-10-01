# Decision log

A record of the choices made building this project, why they were made, and what
was rejected. Kept because the reasoning behind a modelling choice is worth more
than the choice itself, and because three weeks later I will not remember.

---

## 2026-10-01  Undocumented values in the repayment status columns

**Decision.** Leave PAY_0 through PAY_6 exactly as they are.

**The problem.** The UCI documentation describes these columns as: minus one means
paid duly, 1 means one month late, through to 8 meaning eight or more months late.
The data also contains minus two (2,759 rows) and zero (14,737 rows). Zero is the
single most common value in the column, appearing in nearly half the dataset, and
neither value appears in the documentation at all.

**Why leave them.** Three reasons.

The common interpretation, that minus two means no balance due and zero means
revolving credit with the minimum met, is inference rather than documentation.
Collapsing those values into a single "not delinquent" category would mean acting
on an assumption I cannot verify, and if zero is in fact distinct from paid in
full I would be discarding signal from half the rows.

A tree based model splits on values rather than on meanings. If minus two, minus
one and zero behave differently, gradient boosting will find that and use it
without needing an interpretation from me.

It is the more defensible position under questioning. Saying I left undocumented
values intact because I could not verify their meaning is stronger than saying I
assumed they meant something, which invites a question I cannot answer.

**Rejected.** Collapsing minus two, minus one and zero into one category, which
assumes knowledge I do not have. Treating them as missing, which is clearly wrong
given how systematically they appear.

**Caveat worth noting.** This is defensible because the model driving the final
decision is tree based. A linear model would treat these codes as a numeric scale
in which minus two is less than minus one is less than zero, which is meaningless
ordering. If logistic regression were the production model rather than a baseline,
these columns would need one hot encoding first.

---

## 2026-10-01  Excluding the protected attributes

**Decision.** Drop SEX, EDUCATION, MARRIAGE and AGE before modelling. The model
uses 19 behavioural and account level features.

**Why.** In the United States the Equal Credit Opportunity Act and Regulation B
prohibit discrimination on the basis of race, colour, religion, national origin,
sex, marital status, age and receipt of public assistance in any aspect of a
credit transaction. Three of those four columns are directly named. A lender
could not use them, so a model built to represent a lending decision should not
either, regardless of what they add to the metric.

**What the exclusion costs.** Measured rather than assumed. A logistic regression
using only those four columns scores 0.5428 AUC against 0.5000 for a coin flip.
They carry almost no standalone predictive power on this dataset, so the legal
constraint is close to free here. That is worth knowing, because it will not be
true of every dataset and the honest version of this argument requires the number.

**Limitation I want to state rather than hide.** Dropping the columns is necessary
and not sufficient. A model can still proxy for a protected class through
correlated features, and ECOA covers disparate impact as well as direct use. A
production version of this would need fair lending testing on the model output,
not just an exclusion list on the input.

**Rejected.** Keeping them because they are in the dataset and improve the metric.
That optimises for the wrong thing: a model that cannot be deployed is not a
better model.

---

## 2026-10-01  Model choice, and why calibration decided it rather than AUC

**Decision.** Use gradient boosting (HistGradientBoostingClassifier) as the model
the threshold analysis runs on. Keep logistic regression as a reported baseline
rather than deleting it.

**Headline numbers.**

| Model | AUC | Brier | Worst decile gap |
|---|---|---|---|
| Logistic regression | 0.7351 | 0.1427 | 0.132 |
| Gradient boosting | 0.7989 | 0.1299 | 0.040 |

**Why the calibration mattered more than the AUC.** The decile table showed two
problems with logistic regression that AUC alone would not have revealed.

It understated risk by 13 points in the top decile, predicting 57 percent where
71 percent actually defaulted. That is the most expensive place in the
distribution to be optimistic, because it is where the losses are.

And the observed default rate was not monotonic across deciles. It rose, fell
back across deciles 4 to 6, then climbed. Deciles 4 to 6 were also overstated by
7 to 10 points, meaning the model would decline accounts that would have paid.
The middle band is where most of the portfolio sits, so being unreliable there is
not a minor defect.

Gradient boosting produced a monotonic observed column across all ten deciles and
cut the worst gap from 0.132 to 0.040, still slightly optimistic at the top but
conservative almost everywhere else, which is the safer direction for a lender.

**Why this is the real justification.** The threshold analysis in step 4 sets a
cutoff on predicted probability. A cutoff is a statement about probability, so it
is only as good as the calibration. Choosing the model on AUC alone would have
been choosing on the wrong criterion for the decision being made.

**Tuning discipline.** Hyperparameters left near defaults with early stopping on a
15 percent internal validation split. The test set was not used for tuning, so the
reported numbers are not inflated by selection against them.

**Note for anyone reading the decile tables.** Group sizes are not exactly equal
because many accounts receive identical predicted probabilities and qcut cannot
split ties across a bin boundary. Harmless, but it explains counts of 737 and 763
where 750 is expected.

---

## 2026-10-01  Exposure definition, and what decision is being modelled

**Decision.** Exposure is BILL_AMT1, the most recent statement balance, clipped at
zero. Not the credit limit.

**Why.** These are existing accounts with six months of history, so this is not an
origination decision where nothing is known about behaviour. It is a continued
exposure decision: whether to keep extending credit on an account already held, or
act on it by reducing the limit, tightening terms or routing to collections. What
is at risk is the balance outstanding, not the limit granted. Clipped at zero
because some balances are negative, meaning the customer is in credit.

**Rejected.** Using LIMIT_BAL as exposure, which would overstate the loss on every
account that is not fully drawn, and would misdescribe the decision being made.

---

## 2026-10-01  Reporting a threshold band rather than a point

**Decision.** Report the operating range as 0.085 to 0.105 rather than quoting the
single empirical optimum of 0.085. Also report 0.170 as the point above which the
book stops making money at all.

**Why.** The profit curve is broad and bumpy near the top. On 7,500 accounts the
exact peak is sampling noise, and two local humps sit within the plateau. Every
threshold between 0.085 and 0.105 captures at least 95 percent of the available
profit, so claiming precision beyond that band would be false.

**A result worth noting.** The theoretical break even from the economics alone,
margin divided by margin plus loss given default, is 0.0964. That sits inside the
empirical band. Theory and data agree once the comparison is made honestly,
between a point and a range rather than between two points.

**The asymmetry.** Profit falls away far more steeply on the loose side than the
tight side: from break even at 0.17 down to a loss of 38.5 million at the top of
the range. Being slightly too conservative is cheap; being too loose is not. Worth
stating because it argues for operating at the lower end of the band.

---

## 2026-10-01  Sensitivity: which assumption the decision is exposed to

**Decision.** Report the sensitivity grid alongside the headline threshold rather
than defending the assumptions.

**Result.** Across margins of 4 to 14 percent and loss given default of 55 to 95
percent, varying margin moves the optimal threshold by up to 0.135 and varying
loss given default moves it by up to 0.090. Margin is roughly one and a half times
more consequential over those ranges.

**Why, mathematically.** The threshold is m / (m + L). Its sensitivity to margin is
L / (m + L)^2 and to loss is m / (m + L)^2. Since L is about nine times larger than
m under the base assumptions, a one point change in margin shifts the threshold
about nine times as much as a one point change in loss. The observed ratio is only
1.5 because the margin range tested spans 10 points while the loss range spans 40.
The conclusion is therefore dependent on the ranges chosen and those should always
be stated with it.

**Why it is interesting.** Credit risk teams invest heavily in refining loss given
default: recovery rates, collections performance, collateral valuation. The margin
assumption typically belongs to pricing or finance and arrives as a single number
that nobody interrogates. This says the number nobody owns moves the decision more
than the number everyone works on.

---
