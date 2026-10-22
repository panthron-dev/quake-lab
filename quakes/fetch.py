"""Task 1: fetch the USGS GeoJSON feed and turn it into a DataFrame."""
import pandas as pd
import requests

BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"


def fetch_feed(feed: str = "all_week") -> dict:
    """GET {BASE}/{feed}.geojson and return the parsed dict.

    Use a timeout and raise_for_status(). A misspelled feed name returns
    HTTP 200 with a plain-text body, so raise a clear error if the body
    is not valid JSON.
    """
    url = f"{BASE}/{feed}.geojson"
    # timeout: without it, a dead server makes the program hang forever.
    resp = requests.get(url, timeout=15)
    # Turns 4xx/5xx status codes into an exception (HTTPError).
    resp.raise_for_status()
    # USGS quirk: a wrong feed name still returns 200, but the body is the text
    # "404 File Not Found", so .json() fails. Re-raise with a useful message.
    try:
        payload = resp.json()
    except ValueError as exc:
        raise ValueError(
            f"Feed '{feed}' did not return JSON (is the name right?). "
            f"Body starts with: {resp.text[:60]!r}"
        ) from exc
    if "features" not in payload:
        raise ValueError(f"Feed '{feed}' returned JSON without a 'features' list")
    return payload


def geojson_to_df(payload: dict) -> pd.DataFrame:
    """One row per event.

    Columns: 'id' (top level of each feature), every key in 'properties',
    plus 'lon', 'lat', 'depth_km' from geometry.coordinates = [lon, lat, depth].
    """
    rows = []
    for feat in payload.get("features", []):
        # Start with the id, then copy every property (mag, place, time, ...).
        row = {"id": feat.get("id")}
        row.update(feat.get("properties") or {})
        # Coordinates come as a list [lon, lat, depth]. Note the order:
        # GeoJSON puts longitude FIRST, which trips up many people.
        coords = (feat.get("geometry") or {}).get("coordinates") or [None, None, None]
        coords = list(coords) + [None] * (3 - len(coords))
        row["lon"], row["lat"], row["depth_km"] = coords[:3]
        rows.append(row)
    # A list of dicts -> DataFrame. Keys missing from some rows become NaN.
    return pd.DataFrame(rows)


def load_geojson_file(path) -> pd.DataFrame:
    """Helper: read a saved .geojson file (the offline fallback)."""
    import json
    with open(path, encoding="utf-8") as fh:
        return geojson_to_df(json.load(fh))
