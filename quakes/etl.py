"""Part 2, Task A: extract, transform, load, with logging.

    python -m quakes.etl --mode batch     # feed only
    python -m quakes.etl --mode merge     # feed + data/stream/*.jsonl
"""
import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from . import config
from .cleaning import clean, dedupe_latest
from .features import build_features
from .fetch import fetch_feed, geojson_to_df, load_geojson_file

log = logging.getLogger("quakes.etl")
DB_PATH = config.PROCESSED_DIR / "quakes.db"
LOG_PATH = config.ROOT / "logs" / "etl.log"


def extract(feed: str = "all_week") -> pd.DataFrame:
    """Fetch the feed as a DataFrame. If the network fails, log a warning and
    use config.FALLBACK_PATH instead."""
    try:
        df = geojson_to_df(fetch_feed(feed))
        log.info("extract: %d rows from feed %s", len(df), feed)
    except Exception as exc:
        log.warning("extract: network failed (%s); using %s", exc, config.FALLBACK_PATH)
        df = load_geojson_file(config.FALLBACK_PATH)
        log.info("extract: %d rows from fallback file", len(df))
    return df


def read_stream_files() -> pd.DataFrame:
    """Concatenate every data/stream/*.jsonl file (empty DataFrame if none)."""
    frames = [pd.read_json(f, lines=True, convert_dates=False, dtype=False)
              for f in sorted(config.STREAM_DIR.glob("*.jsonl"))
              if f.stat().st_size > 0]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("read_stream_files: %d rows from %d files", len(df), len(frames))
    return df


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Day 1 steps in one call: clean, time/quality/location features, target,
    group_rare on 'region', drop leaky columns."""
    cleaned = clean(raw)
    log.info("transform: %d rows -> %d after cleaning", len(raw), len(cleaned))
    out = build_features(cleaned)
    log.info("transform: %d columns after features and leak removal", out.shape[1])
    return out


def _to_sql_ready(df: pd.DataFrame) -> pd.DataFrame:
    """SQLite has no datetime type: store timestamps as ISO text."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    return df


def load_sqlite(df: pd.DataFrame, db_path, table: str = "quakes") -> None:
    """Upsert on 'id': running twice must not duplicate rows, and a row with an
    existing id must be replaced by the new version.

    Method: DELETE the ids we are about to write, then append them.
    Both happen in one transaction, so a crash can't leave half a load.
    """
    df = _to_sql_ready(df)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        with con:  # commits on success, rolls back on error
            exists = con.execute(
                "select 1 from sqlite_master where type='table' and name=?", (table,)
            ).fetchone()
            if exists:
                # Add any column the table doesn't have yet (schema drift).
                have = {r[1] for r in con.execute(f'pragma table_info("{table}")')}
                for col in df.columns:
                    if col not in have:
                        con.execute(f'alter table "{table}" add column "{col}"')
                ids = df["id"].tolist()
                for i in range(0, len(ids), 500):  # SQLite caps ? placeholders
                    chunk = ids[i:i + 500]
                    marks = ",".join("?" * len(chunk))
                    con.execute(f'delete from "{table}" where id in ({marks})', chunk)
            df.to_sql(table, con, if_exists="append", index=False)
        log.info("load: upserted %d rows into %s.%s", len(df), Path(db_path).name, table)
    finally:
        con.close()


def run_etl(mode: str = "batch") -> None:
    """extract -> (merge stream files if mode == 'merge') -> transform -> load.
    Log row counts at every stage with the logging module."""
    raw = extract()
    if mode == "merge":
        stream = read_stream_files()
        raw = dedupe_latest(pd.concat([raw, stream], ignore_index=True))
        log.info("merge: %d rows after merging stream and dedupe", len(raw))
    data = transform(raw)
    load_sqlite(data, DB_PATH)
    total = sqlite3.connect(DB_PATH).execute("select count(*) from quakes").fetchone()[0]
    log.info("done: table now holds %d rows (positive rate %.2f%%)",
             total, 100 * data[config.TARGET].mean())


def main() -> None:
    """argparse: --mode {batch, merge}. Configure logging to LOG_PATH and the
    console, then call run_etl."""
    p = argparse.ArgumentParser(description="USGS quake ETL into SQLite")
    p.add_argument("--mode", choices=["batch", "merge"], default="batch")
    a = p.parse_args()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )
    run_etl(a.mode)


if __name__ == "__main__":
    main()
