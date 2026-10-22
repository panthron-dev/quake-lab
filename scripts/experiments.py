"""Reproduce every number quoted in ANSWERS.md and ANSWERS_ML.md.

    python scripts/experiments.py            # uses the live feed, else data/fallback
Writes reports/experiments.json and prints a readable summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from quakes import config  # noqa: E402
from quakes import cleaning as C  # noqa: E402
from quakes.etl import extract, read_stream_files, transform  # noqa: E402
from quakes.models import GDLinearRegression  # noqa: E402
from quakes.train import run  # noqa: E402

out = {}

# ---------------------------------------------------------------- Day 1
raw = extract()
stream = read_stream_files()
both = pd.concat([raw, stream], ignore_index=True) if len(stream) else raw
out["raw_rows"] = len(raw)
out["stream_rows"] = len(stream)
out["feed_generated_utc"] = str(pd.to_datetime(raw["time"].max(), unit="ms", utc=True))
out["missing_frac_top"] = raw.isna().mean().sort_values(ascending=False).head(8).round(3).to_dict()

steps = {}
d = C.epoch_ms_to_datetime(both); steps["start"] = len(d)
d = C.dedupe_latest(d); steps["after_dedupe"] = len(d)
out["types_before_filter"] = d["type"].str.strip().str.lower().value_counts().to_dict()
d = C.keep_earthquakes(d); steps["after_keep_earthquakes"] = len(d)
d = C.drop_missing_target(d); steps["after_drop_missing_mag"] = len(d)
out["shape_steps"] = steps
if len(stream):
    ids = both.groupby("id")["updated"].nunique()
    out["stream_changed_events"] = int((ids > 1).sum())
    out["stream_unique_ids"] = int(stream["id"].nunique())

out["neg_depth"] = int((d["depth_km"] < 0).sum())
out["neg_depth_min"] = float(d["depth_km"].min())
out["neg_mag"] = int((d["mag"] < 0).sum())
out["neg_mag_min"] = float(d["mag"].min())
out["iqr_flagged_depth"] = int(C.iqr_outlier_mask(d["depth_km"]).sum())
out["iqr_flagged_mag"] = int(C.iqr_outlier_mask(d["mag"]).sum())
out["neg_depth_nets"] = d.loc[d["depth_km"] < 0, "net"].value_counts().head(5).to_dict()
out["neg_mag_nets"] = d.loc[d["mag"] < 0, "net"].value_counts().head(5).to_dict()

region = C.extract_region(d["place"])
out["n_regions"] = int(region.nunique())
out["regions_seen_once"] = int((region.value_counts() == 1).sum())
out["top15_coverage"] = float(region.value_counts().head(15).sum() / len(region))
big = (d["mag"] >= 4.5).astype(int)
out["big_rate"] = float(big.mean())
out["magType_big_rate"] = (pd.DataFrame({"m": d["magType"], "b": big})
                           .groupby("m")["b"].agg(["mean", "count"]).round(3)
                           .sort_values("count", ascending=False).head(8).to_dict("index"))
out["tsunami_big_rate"] = (pd.DataFrame({"t": d["tsunami"], "b": big})
                           .groupby("t")["b"].agg(["mean", "count"]).round(3).to_dict("index"))

# ---------------------------------------------------------------- Day 2
data = transform(both)
out["model_rows"] = len(data)

reg_lin = run("regression", "linear", df=data, save=False)
out["Q1_regression"] = {"baseline": reg_lin["baseline_test"], "linear": reg_lin["test"],
                        "cv_r2": reg_lin["cv_r2_train"]}

clf = run("classification", "logistic", df=data, save=False)
clf_bal = run("classification", "logistic", balanced=True, df=data, save=False)
out["Q2_dummy"] = clf["baseline_test"]
out["Q3_threshold"] = {"chosen": clf["threshold"], "at_0.5": clf["test_at_0.5"],
                       "at_chosen": clf["test_at_threshold"],
                       "balanced_threshold": clf_bal["threshold"],
                       "balanced_at_0.5": clf_bal["test_at_0.5"],
                       "balanced_at_chosen": clf_bal["test_at_threshold"]}

# Q4: un-drop a leaky column. NUMERIC is a whitelist, so removing it from LEAKY
# alone changes nothing; it must also be fed to the model.
leak = {}
orig_leaky, orig_num = list(config.LEAKY), list(config.NUMERIC)
try:
    config.LEAKY[:] = [c for c in orig_leaky if c != "sig"]
    d_leak = transform(both)
    leak["only_removed_from_LEAKY"] = run("classification", "logistic", df=d_leak, save=False)["test_at_threshold"]
    config.NUMERIC[:] = orig_num + ["sig"]
    leak["sig_used_as_feature"] = run("classification", "logistic", df=d_leak, save=False)["test_at_threshold"]
    leak["sig_used_regression"] = run("regression", "linear", df=d_leak, save=False)["test"]
finally:
    config.LEAKY[:], config.NUMERIC[:] = orig_leaky, orig_num
leak["clean_classification"] = clf["test_at_threshold"]
leak["clean_regression"] = reg_lin["test"]
out["Q4_leakage"] = leak

out["Q5_poly"] = {}
for deg in (1, 2, 4):
    r = run("regression", "poly", degree=deg, alpha=1.0, df=data, save=False)
    out["Q5_poly"][deg] = {"train": r["train"], "test": r["test"]}

# Q6: GD vs closed form on the notebook's synthetic data
rng = np.random.default_rng(42)
X_syn = rng.normal(size=(500, 3))
y_syn = X_syn @ np.array([2.0, -1.0, 0.5]) + 3.0 + rng.normal(0, 0.5, 500)
gd = GDLinearRegression(lr=0.1, n_iter=1000).fit(X_syn, y_syn)
ref = LinearRegression().fit(X_syn, y_syn)
X_raw = X_syn * np.array([1.0, 50.0, 0.1])
y_raw = X_raw @ np.array([2.0, -0.04, 10.0]) + 3.0 + rng.normal(0, 0.5, 500)
gd_raw = GDLinearRegression(lr=1e-4, n_iter=500).fit(X_raw, y_raw)
gd_sc = GDLinearRegression(lr=0.1, n_iter=500).fit(StandardScaler().fit_transform(X_raw), y_raw)
ref_raw = LinearRegression().fit(X_raw, y_raw)
with np.errstate(all="ignore"):
    gd_blow = GDLinearRegression(lr=1e-3, n_iter=500).fit(X_raw, y_raw)
out["Q6_gd"] = {
    "gd_coef": gd.coef_.round(4).tolist(), "sk_coef": ref.coef_.round(4).tolist(),
    "gd_intercept": round(gd.intercept_, 4), "sk_intercept": round(float(ref.intercept_), 4),
    "unscaled_final_loss": round(gd_raw.loss_history_[-1], 3),
    "scaled_final_loss": round(gd_sc.loss_history_[-1], 3),
    "unscaled_coef": gd_raw.coef_.round(3).tolist(), "true_raw_coef": ref_raw.coef_.round(3).tolist(),
    "unscaled_lr1e-3_final_loss": float(gd_blow.loss_history_[-1]),
}

(config.ROOT / "reports").mkdir(exist_ok=True)
(config.ROOT / "reports" / "experiments.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))
