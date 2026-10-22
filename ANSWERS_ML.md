# Day 2 Lab: Answers

Name: Nouman Jaffar

Data: USGS `all_week` feed generated 2026-10-22 05:35 UTC, 1894 earthquakes after the ETL
(1515 train / 379 test, stratified, seed 42). Every number below is reproduced by
`python scripts/experiments.py` (saved to `reports/experiments.json`).

## 1. Baseline
How much does your regression beat "predict the mean" (RMSE and R²)? Is it worth deploying?

| Test set | MAE | RMSE | R² |
|---|---|---|---|
| Dummy (predict the mean) | 0.965 | 1.314 | −0.000 |
| Linear regression | 0.386 | 0.508 | 0.850 |

The linear model cuts RMSE by about 61 % (1.31 → 0.51) and lifts R² from 0 to 0.85, and 5-fold
CV on train agrees (R² 0.859 ± 0.007), so it isn't a lucky split.
**I would not deploy it as a magnitude predictor anyway.** Most of the R² comes from *which
network reported the quake*, not from physics. Outside the US the feed only lists bigger
quakes (`net = "us"` has a mean magnitude of 4.49, against about 1–2 for local networks such as
`ak`, `nc` and `ci`), so `net` and `region` mainly encode USGS's reporting rules. It's a
useful sanity check on this feed, but it would not carry over to a catalog with different
reporting. An error of ±0.5 magnitude is also about a 3× error in shaking amplitude.

## 2. Accuracy paradox
What accuracy and recall does the dummy classifier get on `big_quake`? Why is accuracy the
wrong headline metric?

The dummy classifier always predicts "not big". It gets **accuracy 0.937 and recall 0.0**
(precision 0, F1 0, ROC-AUC 0.5). Only 6.3 % of quakes are big, so a model that never raises
an alarm is right 93.7 % of the time and still misses every big quake, which is the whole
point of the system. Accuracy is dominated by the easy majority class. Recall, precision, F1
and ROC-AUC show how well the model handles the rare class we actually care about.

## 3. Threshold
What threshold did your search choose, compared with 0.5? Which error would you rather make in
an earthquake alert system, and why?

The search on **out-of-fold training probabilities** (`cross_val_predict`, so the test set was
never used to choose it) picked **0.35**. On the test set (24 big quakes):

| Threshold | Recall | Precision | F1 |
|---|---|---|---|
| 0.50 | 0.833 (20 of 24 caught) | 0.769 | 0.800 |
| **0.35** | **0.875 (21 of 24 caught)** | 0.750 | 0.808 |

Lowering the threshold catches one more big quake in exchange for a little precision.
**In an alert system I'd rather make false alarms (false positives) than miss a big quake
(false negatives).** A missed alert can cost lives and damage, while a false alarm costs
inconvenience. The limit is alert fatigue: if alarms are false too often, people stop
reacting, so precision can't be ignored completely. (With `--balanced` class weights recall
reaches 0.958 at precision 0.72.)

## 4. Leakage check
Remove one leaky column name from `config.LEAKY`, rerun the ETL and train, and compare scores.
What happened, and why is the better number a lie?

I removed `sig` from `LEAKY`. **At first nothing changed** (F1 still 0.808), because
`config.NUMERIC` is a whitelist: a column the preprocessor isn't told about never reaches
the model. Once I also added `sig` to `NUMERIC`:

| | Clean | With `sig` |
|---|---|---|
| Classification F1 | 0.808 | **0.936** |
| Classification recall | 0.875 | 0.917 |
| ROC-AUC | 0.9948 | 0.9995 |
| Regression R² | 0.850 | **0.936** |

The scores jump, and **that better number is a lie**: `sig` (significance) is calculated by
USGS *from the magnitude* (plus felt reports) after the event. The model isn't predicting the
magnitude, it's reading it back from a column computed from it. In production, at the moment
of prediction, `sig` doesn't exist yet, so real performance would fall back to the clean
numbers or lower. A score that looks almost perfect on a hard problem is a warning sign.

## 5. Complexity
Train with `--model poly` at degree 1, 2 and 4. When does performance on the test set fall below
the training result?

| Degree | Train R² | Test R² | Test RMSE |
|---|---|---|---|
| 1 | 0.868 | 0.851 | 0.507 |
| 2 | 0.907 | **0.885** | **0.445** |
| 4 | 0.967 | **−6014** | 101.9 |

Test R² is a little below train at every degree, which is the normal generalisation gap. At
degrees 1 and 2 the gap is small (about 0.02), and degree 2 is the best model on test. **At
degree 4 the test result collapses** even though train R² keeps rising. The 14 numeric columns
expand into thousands of polynomial terms, the model fits noise, and on test rows slightly
outside the training range those high powers blow up (some predicted magnitudes are wildly
wrong, so RMSE is about 102). That's overfitting: training error alone would have picked the
worst model.

## 6. Gradient descent
Do your `GDLinearRegression` coefficients match `LinearRegression` on the notebook data? What
changes if you skip scaling?

**Yes, to 4 decimals.** On standardized data (lr = 0.1, 1000 iterations):

| | w₁ | w₂ | w₃ | b |
|---|---|---|---|---|
| GDLinearRegression | 1.9557 | −0.9889 | 0.5195 | 2.9217 |
| LinearRegression | 1.9557 | −0.9889 | 0.5195 | 2.9217 |

**Skipping scaling** (features on scales of about 1, 50 and 0.1): the largest feature controls
how big a step can be. With lr = 1e-3 the loss **diverged** to about 10¹⁴⁹. To stay stable I had
to drop to lr = 1e-4, and then after 500 iterations the loss was still 6.57 (against 0.13 when
scaled). The coefficients were nowhere near the answer: [0.107, −0.037, 0.004] against the true
[2.01, −0.04, 10.03]. The small-scale feature gets tiny gradients and barely moves. Scaling
gives the loss surface a round bowl instead of a long narrow valley, so one learning rate works
for every weight.
