"""Task 4: feature engineering."""
import pandas as pd

from . import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'hour' and 'dayofweek' (UTC) from 'time'."""
    df = df.copy()
    t = df["time"].dt.tz_convert("UTC")
    df["hour"] = t.dt.hour
    df["dayofweek"] = t.dt.dayofweek  # Monday = 0 ... Sunday = 6
    return df


def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add:
    update_lag_hours = hours between 'time' and 'updated'
    is_reviewed      = 1 if status == 'reviewed' else 0
    nst_missing      = 1 if 'nst' is missing else 0
    """
    df = df.copy()
    df["update_lag_hours"] = (df["updated"] - df["time"]).dt.total_seconds() / 3600
    df["is_reviewed"] = (df["status"] == "reviewed").astype(int)
    # "Missing" can itself be information: some networks never report nst.
    df["nst_missing"] = df["nst"].isna().astype(int)
    return df


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'abs_lat' and 'is_shallow' (1 if depth_km < 70 else 0)."""
    df = df.copy()
    df["abs_lat"] = df["lat"].abs()  # distance from the equator, N or S
    df["is_shallow"] = (df["depth_km"] < 70).astype(int)
    return df


def group_rare(s: pd.Series, top_k: int = 15) -> pd.Series:
    """Keep the top_k most frequent values; replace all others with 'Other'."""
    # Without this, one-hot encoding 'region' makes one column per region
    # (hundreds), most of them seen only once or twice: noise, not signal.
    top = s.value_counts().nlargest(top_k).index
    return s.where(s.isin(top), "Other")


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'big_quake' = 1 if mag >= 4.5 else 0."""
    df = df.copy()
    df[config.TARGET] = (df["mag"] >= 4.5).astype(int)
    return df


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every column listed in config.LEAKY (ignore ones that are absent)."""
    return df.drop(columns=[c for c in config.LEAKY if c in df.columns])


def build_features(df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """All Task 4 steps in order, on an already-cleaned DataFrame."""
    df = add_time_features(df)
    df = add_quality_features(df)
    df = add_location_features(df)
    df = add_target(df)
    df["region"] = group_rare(df["region"], top_k=top_k)
    return drop_leaky_columns(df)


def events_last_24h_in_region(df: pd.DataFrame) -> pd.Series:
    """Stretch: how many earlier quakes hit the same region in the previous 24 h.

    Uses only PAST events (closed='left' excludes the current row), so it is
    something you would really know at prediction time.
    """
    s = pd.Series(1.0, index=df.index)
    out = pd.Series(0.0, index=df.index)
    for _, idx in df.groupby("region").groups.items():
        sub = df.loc[idx].sort_values("time")
        counts = (s.loc[sub.index].set_axis(sub["time"])
                   .rolling("24h", closed="left").sum().fillna(0).to_numpy())
        out.loc[sub.index] = counts
    return out.astype(int)
