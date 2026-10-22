"""Task 2: simulate streaming by polling a feed on a timer."""
import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config
from .fetch import fetch_feed, geojson_to_df


def filter_unseen(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Return only rows whose (id, updated) pair is not in `seen`.

    Add the new pairs to `seen` (modify the set in place).
    """
    # Why (id, updated) and not just id? The same quake is re-published when
    # USGS revises it (new magnitude, reviewed status...). A new `updated`
    # value means a NEW VERSION of the event, which we want to keep.
    keys = list(zip(df["id"], df["updated"]))
    mask = [k not in seen for k in keys]
    seen.update(k for k, new in zip(keys, mask) if new)
    return df.loc[mask]


def run_stream(feed: str, interval_s: int, duration_s: int, out_path: str) -> None:
    """Poll `feed` every `interval_s` seconds for `duration_s` seconds.

    Each poll: fetch, parse, keep unseen rows, append them to `out_path`
    as JSON lines, and print e.g. "[14:02:11] 3 new / 12 fetched".
    A failed poll must not stop the loop.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        stamp = datetime.now().strftime("%H:%M:%S")
        try:
            df = geojson_to_df(fetch_feed(feed))
            new = filter_unseen(df, seen)
            if len(new):
                # "a" = append mode, so each poll adds to the file instead of
                # overwriting it. JSON Lines = one JSON object per line.
                with open(out_path, "a", encoding="utf-8") as fh:
                    new.to_json(fh, orient="records", lines=True)
                    fh.write("\n")
            print(f"[{stamp}] {len(new)} new / {len(df)} fetched", flush=True)
        except Exception as exc:  # network blips must not kill a 40-minute run
            print(f"[{stamp}] poll failed: {exc}", flush=True)
        time.sleep(interval_s)


def main() -> None:
    """argparse: --feed (default all_hour), --interval (60), --duration (2400),
    --out (data/stream/stream.jsonl), then call run_stream."""
    p = argparse.ArgumentParser(description="Poll a USGS feed and save new events.")
    p.add_argument("--feed", default="all_hour")
    p.add_argument("--interval", type=int, default=60, help="seconds between polls")
    p.add_argument("--duration", type=int, default=2400, help="total seconds to run")
    p.add_argument("--out", default=str(config.STREAM_DIR / "stream.jsonl"))
    a = p.parse_args()
    run_stream(a.feed, a.interval, a.duration, a.out)


if __name__ == "__main__":
    main()
