# Learning Guide: how this project works and why

Read this next to the code. Each section has: **the idea**, **the key lines**, **the trap
the lab is teaching you to avoid**, and **check yourself** questions. Try to answer those
without looking. If you can explain them out loud, you understand the lab.

---

## The big picture

```
USGS feed ──fetch──▶ raw DataFrame ──clean──▶ one row per real quake
   ▲                     ▲                         │
   │ every 60 s          │ merge                   ▼
 stream.py ─────▶ data/stream/*.jsonl        features + target ──drop leaky──▶ model-ready
                                                                      │
                                    split (train/test) ◀──────────────┘
                                       │
                       fit preprocessor on TRAIN only ──▶ transform test
                                       │
                            Day 2: etl.py → SQLite → train.py → models/ + reports/
```

The whole lab is built around one question: **"Would I know this at the moment I predict?"**
Every bug the lab teaches (leakage, duplicates, fitting on test) is a way of
accidentally answering "yes" when the truth is "no".

---

## Task 1: fetch (`quakes/fetch.py`)

**Idea.** An API gives you nested JSON; a model needs a flat table. You flatten each
`feature` into one dict, then let pandas build the DataFrame.

**Key lines.**
- `requests.get(url, timeout=15)`: always use a timeout. Without one, a dead server hangs your script forever.
- `resp.raise_for_status()` turns 404/500 into an exception.
- `try: resp.json() except ValueError:` exists because **USGS returns HTTP 200 with a
  plain-text "404" body** for a misspelled feed. Status codes alone can lie.
- `coordinates = [lon, lat, depth]`: GeoJSON puts **longitude first**. That's a classic mix-up.

**Check yourself.** Why build a `list` of dicts and call `pd.DataFrame` once, instead of
appending rows to a DataFrame in the loop? (Answer: appending copies the whole frame
every time, so it's O(n²).)

---

## Task 2: stream (`quakes/stream.py`)

**Idea.** Polling a feed every 60 s simulates a stream. The key is the dedupe key: **`(id, updated)`**.

- `id` alone would ignore *revisions*: USGS re-publishes a quake when its magnitude or
  status changes, keeping the same `id` with a newer `updated`.
- `(id, updated)` keeps each **version** exactly once.
- `seen` is a `set`, so lookups are O(1), and it's modified **in place** so state carries across polls.
- Append mode `"a"` plus JSON Lines means each poll adds lines and never rewrites the file.
- `try/except` **inside** the loop means one failed poll doesn't kill a 40-minute run.

**Check yourself.** What would go wrong if `filter_unseen` used `seen = seen | new` instead of `seen.update(...)`?
(The caller's set wouldn't change, so every poll would look "new" again.)

---

## Task 3: cleaning (`quakes/cleaning.py`)

| Step | Why |
|---|---|
| `epoch_ms_to_datetime` | `time` is milliseconds since 1970. `pd.to_datetime(unit="ms", utc=True)`. Always keep UTC. |
| `dedupe_latest` | `sort_values("updated")` then `drop_duplicates("id", keep="last")` = **latest version wins**. |
| `keep_earthquakes` | `" Earthquake"` ≠ `"earthquake"` until you `strip().lower()`. Explosions and quarry blasts are a different phenomenon. |
| `extract_region` | `"10 km SW of Ridgecrest, CA"` → `"CA"`. `str.split(",").str[-1]`: with no comma, the last piece is the whole string. |
| `iqr_outlier_mask` | IQR = Q3 − Q1. Values beyond 1.5·IQR from the box are *flagged*, **not deleted**. |
| `drop_missing_target` | A row without `mag` has no label, so you can't train or score on it. |

**Order in `clean()`.** Dedupe *before* filtering. If you filtered first, an old version
could survive while its newer version (say, retyped as "explosion") was removed.

**The outlier lesson.** "Outlier" means *unusual*, not *wrong*. Negative depth and
negative magnitude are physically real (see ANSWERS.md Q3). Deleting them would teach
the model a world that doesn't exist.

---

## Task 4: features (`quakes/features.py`, `quakes/config.py`)

- **Time:** `hour`, `dayofweek`. Small signal, but they *are* known at prediction time.
- **Quality:** `nst_missing`, because a *missing* value can itself be informative (some networks never report it).
- **Location:** `abs_lat` (distance from the equator, ignoring N/S), `is_shallow`.
- **`group_rare`:** keep the 15 most common regions and call the rest `"Other"`.
  Without it, one-hot makes one column per region (hundreds), most seen once.
  That gives the model columns it can't learn from, and makes unseen regions more likely at test time.
- **Target:** `big_quake = mag >= 4.5`.

**Leakage (the heart of the lab).** A leaky column is one that is computed *from* the
answer, or only exists *because of* it. Go through the columns:

| Column | Known at prediction time? | Verdict |
|---|---|---|
| `title` | contains "M 4.6 - …" | **leaky** |
| `sig` | computed from mag | **leaky** |
| `mmi`, `alert` | ShakeMap/PAGER products, made for big quakes | **leaky** |
| `felt`, `cdi` | people only feel big quakes | **leaky** |
| `tsunami` | flag set for large oceanic events | **leaky** |
| `magType` | method chosen *by* size (ml small → mww large) | **leaky** |
| `types` | list of products; `shakemap,losspager` = big quake | **leaky** |
| `depth_km`, `lat`, `lon`, `nst`, `gap`, `rms` | from locating the quake | OK |

The giveaway is a model that looks almost perfect. On a hard problem, that usually means leakage.

---

## Task 5: transform + run (`quakes/transform.py`, `quakes/run.py`)

- **Remove `mag` from X.** `big_quake` is defined from `mag`, so keeping `mag` is the most
  extreme leak possible.
- **`stratify=y`.** Big quakes are rare (a few %). Without stratifying, the test set
  might get noticeably more or fewer positives than train, and the scores would be noise.
- **`ColumnTransformer`** applies different steps to different columns:
  - numeric: median impute (robust to skew) → scaler
  - nominal: most-frequent impute → one-hot with `handle_unknown="ignore"` (an unseen region becomes all-zeros instead of crashing)
- **`fit_transform` on train, `transform` on test.** The medians, scaler stats and
  category list are *learned*. Learning them from the test rows leaks test information.
- **`RobustScaler`** uses median/IQR instead of mean/std. Depth has a long tail (a few
  quakes at 600 km), so the robust version isn't dragged around by those few points.

---

## Part 1 notebook: scikit-learn basics

1. **Estimator API:** `fit` learns, `predict` uses it; learned things end in `_`.
2. **Baseline first:** R² 0.45 sounds bad until you see the mean-predictor gets ≈ 0.
3. **Cross-validation:** one split can be lucky; 5 folds give a mean ± spread.
4. **Pipelines prevent leakage:** the scaler is refit inside every fold.
5. **Gradient descent:** `w -= lr * X.T @ err / m`. Learning rate too small = slow,
   too big = divergence. Unscaled features force a tiny learning rate.
6. **Polynomial + Ridge:** more degree = lower train error, but validation error turns
   back up (overfitting). Ridge shrinks weights; `alpha` must be tuned.
7. **Logistic regression:** outputs probabilities; the 0.5 threshold is a choice, not a law.

---

## Part 2: ETL, models, evaluation, training

- **ETL (`etl.py`)**: *Extract* (feed or fallback) → *Transform* (Day 1 steps) → *Load*
  (SQLite). The load is an **upsert**: delete the ids being written, then append, all in one
  transaction. Running it twice gives the same table (**idempotent**), which is what makes
  it safe to schedule every hour.
- **Logging**: `logging` with file + console handlers; row counts at every stage, so a
  bad run is diagnosable after the fact.
- **`GDLinearRegression`**: inherits `BaseEstimator, RegressorMixin`, so it gets
  `get_params`, `score`, and works inside `Pipeline` and `cross_val_score`.
- **Metrics (`evaluate.py`)**: MAE/RMSE/R² for regression; for classification, accuracy is
  misleading on rare classes, so look at precision/recall/F1/ROC-AUC.
- **Threshold (`train.py`)**: chosen from **out-of-fold** training predictions
  (`cross_val_predict`), so the test set is only used once, at the very end.

---

## How to run everything

```bash
conda env create -f environment.yml && conda activate niai-ml
pytest                                    # 23 tests
python -m quakes.stream --interval 60 --duration 2400   # leave running in terminal 2
python -m quakes.run                      # Day 1 report + data/processed/*
python -m quakes.etl --mode merge         # Day 2: SQLite
python -m quakes.train --task regression --model linear
python -m quakes.train --task classification --model logistic
python scripts/experiments.py             # reproduces every number in ANSWERS_ML.md
```
