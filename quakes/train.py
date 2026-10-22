"""Part 2, Task D: train a model on the ETL output and save it.

Command line:
    python -m quakes.train --task regression --model linear
    python -m quakes.train --task regression --model poly --degree 3 --alpha 10
    python -m quakes.train --task classification --model logistic [--balanced]

Steps (see the lab handout):
 1. load the table from the SQLite database written by quakes.etl
 2. X = everything except 'mag' and 'big_quake'; y = 'mag' or 'big_quake'
 3. train/test split (stratified for classification, seeded with config.SEED)
 4. fit the dummy baseline AND your model; report both on the test set
 5. cross-validate on the TRAIN set only
 6. classification only: pick the threshold from out-of-fold training
    predictions (cross_val_predict, method='predict_proba'), then apply it to test
 7. save the pipeline to models/ and the report (JSON) to reports/
"""
import argparse
import json
import sqlite3

import joblib
import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split

from . import config
from .etl import DB_PATH
from .evaluate import (best_threshold, classification_report_dict,
                       cross_validate_model, regression_report)
from .models import make_classifier, make_regressor


def load_data(db_path=None):
    con = sqlite3.connect(db_path or DB_PATH)
    try:
        return pd.read_sql("select * from quakes", con)
    finally:
        con.close()


def run(task, model, degree=2, alpha=1.0, C=1.0, balanced=False, df=None, save=True):
    """Train + evaluate; returns the report dict (also used by experiments)."""
    df = load_data() if df is None else df
    X = df.drop(columns=["mag", config.TARGET])
    y = df["mag"] if task == "regression" else df[config.TARGET]
    strat = y if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=strat, random_state=config.SEED)

    report = {"task": task, "model": model, "n_train": len(X_tr), "n_test": len(X_te)}
    if task == "regression":
        base = make_regressor("dummy").fit(X_tr, y_tr)
        pipe = make_regressor(model, degree=degree, alpha=alpha).fit(X_tr, y_tr)
        if model == "poly":
            report["degree"] = degree
        if model in ("poly", "ridge"):
            report["alpha"] = alpha
        report["baseline_test"] = regression_report(y_te, base.predict(X_te))
        report["train"] = regression_report(y_tr, pipe.predict(X_tr))
        report["test"] = regression_report(y_te, pipe.predict(X_te))
        report["cv_r2_train"] = cross_validate_model(
            make_regressor(model, degree=degree, alpha=alpha), X_tr, y_tr, "r2")
    else:
        cw = "balanced" if balanced else None
        base = make_classifier("dummy").fit(X_tr, y_tr)
        pipe = make_classifier(model, C=C, class_weight=cw).fit(X_tr, y_tr)
        report["balanced"] = balanced
        report["baseline_test"] = classification_report_dict(
            y_te, base.predict(X_te), base.predict_proba(X_te)[:, 1])
        p_te = pipe.predict_proba(X_te)[:, 1]
        report["test_at_0.5"] = classification_report_dict(y_te, (p_te >= 0.5).astype(int), p_te)
        report["cv_roc_auc_train"] = cross_validate_model(
            make_classifier(model, C=C, class_weight=cw), X_tr, y_tr, "roc_auc")
        # Threshold chosen from OUT-OF-FOLD train predictions: each training row
        # is scored by a model that never saw it, so the test set stays untouched.
        oof = cross_val_predict(make_classifier(model, C=C, class_weight=cw),
                                X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
        t = best_threshold(y_tr, oof, "f1")
        report["threshold"] = t
        report["test_at_threshold"] = classification_report_dict(
            y_te, (p_te >= t).astype(int), p_te)

    if save:
        tag = f"{task}_{model}" + (f"_d{degree}" if model == "poly" else "") \
              + ("_balanced" if balanced else "")
        (config.ROOT / "models").mkdir(exist_ok=True)
        (config.ROOT / "reports").mkdir(exist_ok=True)
        joblib.dump(pipe, config.ROOT / "models" / f"{tag}.joblib")
        (config.ROOT / "reports" / f"{tag}.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Train a quake model")
    p.add_argument("--task", choices=["regression", "classification"], required=True)
    p.add_argument("--model", default="linear",
                   help="regression: dummy|linear|ridge|poly|gd; classification: dummy|logistic")
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--balanced", action="store_true", help="class_weight='balanced'")
    a = p.parse_args()
    report = run(a.task, a.model, a.degree, a.alpha, a.C, a.balanced)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
