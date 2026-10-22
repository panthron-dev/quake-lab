"""Task 5: run the whole pipeline end to end.

Steps (see LAB_GUIDE.md):
 1. fetch all_week (fall back to config.FALLBACK_PATH if the network fails)
 2. merge any data/stream/*.jsonl rows, then dedupe_latest
 3. clean, engineer features, drop leaky columns
 4. split, fit_transform on train only, transform test
 5. print the report, save train.csv, test.csv, preprocessor.joblib

    python -m quakes.run [--scaler robust|standard|minmax]
"""
import argparse

import joblib
import pandas as pd

from . import config
from .cleaning import clean
from .features import build_features
from .fetch import fetch_feed, geojson_to_df, load_geojson_file
from .transform import build_preprocessor, split_data


def load_batch(feed: str = "all_week") -> pd.DataFrame:
    try:
        return geojson_to_df(fetch_feed(feed))
    except Exception as exc:
        print(f"Network fetch failed ({exc}); using {config.FALLBACK_PATH.name}")
        return load_geojson_file(config.FALLBACK_PATH)


def load_stream() -> pd.DataFrame:
    files = sorted(config.STREAM_DIR.glob("*.jsonl"))
    frames = [pd.read_json(f, lines=True, convert_dates=False, dtype=False)
              for f in files if f.stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaler", default="robust", choices=["robust", "standard", "minmax"])
    args = ap.parse_args()

    batch = load_batch()
    stream = load_stream()
    raw = pd.concat([batch, stream], ignore_index=True)

    # "Changed" = same id seen with more than one `updated` value.
    changed = int((raw.groupby("id")["updated"].nunique() > 1).sum())

    cleaned = clean(raw)  # clean() does the dedupe_latest
    data = build_features(cleaned)

    X_train, X_test, y_train, y_test = split_data(data)
    pre = build_preprocessor(args.scaler)
    # fit ONLY on train: medians, scaler stats and one-hot categories must
    # not see the test rows, or the test score is no longer honest.
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)

    print("=" * 46)
    print(f"raw rows (batch)             {len(batch):>8}")
    print(f"stream rows merged           {len(stream):>8}")
    print(f"events whose `updated` moved {changed:>8}")
    print(f"rows after cleaning          {len(cleaned):>8}")
    print(f"train matrix                 {str(Xtr.shape):>8}")
    print(f"test matrix                  {str(Xte.shape):>8}")
    print(f"positive rate train / test   {y_train.mean():.3%} / {y_test.mean():.3%}")
    print("=" * 46)

    # Save the un-transformed rows (with mag + big_quake) so Day 3 regression
    # and Day 4 classification can both reuse them, plus the fitted preprocessor.
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["id", "time"] + config.NUMERIC + config.NOMINAL
    for name, X, y in (("train", X_train, y_train), ("test", X_test, y_test)):
        out = X[[c for c in keep if c in X]].copy()
        out["mag"] = data.loc[X.index, "mag"]
        out[config.TARGET] = y
        out.to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)
    joblib.dump(pre, config.PROCESSED_DIR / "preprocessor.joblib")
    print(f"saved train.csv, test.csv, preprocessor.joblib to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
