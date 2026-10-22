"""Task 3: cleaning."""
import pandas as pd


def epoch_ms_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'time' and 'updated' (epoch milliseconds) to UTC datetimes."""
    df = df.copy()  # never mutate the caller's DataFrame
    for col in ("time", "updated"):
        if col in df and not pd.api.types.is_datetime64_any_dtype(df[col]):
            # Epoch ms = milliseconds since 1970-01-01 UTC.
            df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
    return df


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """One row per 'id', keeping the row with the greatest 'updated'."""
    # Sort oldest -> newest, then keep the LAST row per id = newest version.
    return (df.sort_values("updated", kind="stable")
              .drop_duplicates("id", keep="last")
              .reset_index(drop=True))


def keep_earthquakes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise 'type' (strip, lowercase) and keep only 'earthquake'."""
    df = df.copy()
    # " Earthquake" and "earthquake" must count as the same thing.
    df["type"] = df["type"].astype("string").str.strip().str.lower()
    return df[df["type"] == "earthquake"].reset_index(drop=True)


def extract_region(place: pd.Series) -> pd.Series:
    """Text after the last comma, or the whole string if there is no comma.
    Missing places become 'Unknown'."""
    # "10 km SW of Ridgecrest, CA" -> "CA";  "Tonga" -> "Tonga"
    # split(",") with no comma gives a 1-item list, so str[-1] is the whole text.
    region = place.astype("string").str.split(",").str[-1].str.strip()
    return region.fillna("Unknown").replace("", "Unknown").astype(object)


def iqr_outlier_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    """True where a value lies outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1  # spread of the middle 50% of values
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where 'mag' is missing."""
    # We can't learn from (or score) a row whose label is unknown.
    return df.dropna(subset=["mag"]).reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Chain the steps above (think about the order) and add a 'region' column.

    Order matters:
    1. datetimes first, so every later step sees real timestamps;
    2. dedupe BEFORE filtering, so an old version of an event can never
       survive while its newer version is dropped (or vice versa);
    3. keep earthquakes, then drop rows with no target.
    Outliers are NOT removed here: negative depth/magnitude are real
    (see ANSWERS.md, question 3).
    """
    df = epoch_ms_to_datetime(df)
    df = dedupe_latest(df)
    df = keep_earthquakes(df)
    df = drop_missing_target(df)
    df["region"] = extract_region(df["place"])
    return df
