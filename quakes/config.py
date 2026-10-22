from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FALLBACK_PATH = ROOT / "data" / "fallback" / "all_week.geojson"
STREAM_DIR = ROOT / "data" / "stream"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
TARGET = "big_quake"

# Numeric features: things known the moment the quake is LOCATED,
# before (or independent of) its magnitude.
NUMERIC: list[str] = [
    "lat", "lon", "depth_km", "abs_lat", "is_shallow",   # where
    "hour", "dayofweek",                                  # when
    "nst", "gap", "dmin", "rms",                          # how well it was recorded
    "nst_missing", "is_reviewed", "update_lag_hours",
]

# Categorical features (one-hot encoded). region is grouped to top-k first.
NOMINAL: list[str] = ["region", "net"]

# Leaky = derived FROM the magnitude, or only exists BECAUSE of it.
# Test for every column: "would I know this at the moment I predict?"
LEAKY: list[str] = [
    "title",    # literally "M 4.6 - 10 km SW of ..." : contains the answer
    "sig",      # significance score, computed from mag (+ felt reports)
    "mmi",      # max shaking intensity (ShakeMap), produced only for larger quakes
    "cdi",      # "Did You Feel It?" intensity: people feel big quakes
    "felt",     # number of felt reports: same story
    "alert",    # PAGER alert level: computed from expected shaking/losses
    "tsunami",  # flag set only for large oceanic events
    "magType",  # method is chosen BY size (ml/md small, mb/mww large)
    "types",    # list of USGS products; shakemap/losspager exist only for big ones
]
