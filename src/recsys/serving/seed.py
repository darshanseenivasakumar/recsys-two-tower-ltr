"""Load the bundle's snapshot interactions into Redis (run once per bundle).

python -m recsys.serving.seed --bundle artifacts/serving/ml-1m --redis-url redis://...

Deploying a DIFFERENT bundle also clears the monitoring streams: their rows were logged
by the previous model, and /monitoring would otherwise compare them with the new
model's drift reference. Re-seeding the same bundle (e.g. after a restart) keeps them.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from recsys.serving.store import EventLog, InteractionStore


def deploy(store: InteractionStore, history: pd.DataFrame, version: str) -> tuple[int, bool]:
    """Seed `history` for bundle `version`; returns (#events, monitoring log cleared)."""
    new_bundle = store.bundle_version() != version
    if new_bundle:
        EventLog(store.client).clear()
    return store.seed(history, version), new_bundle


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m recsys.serving.seed")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    store = InteractionStore.from_url(args.redis_url)
    start = time.perf_counter()
    n, cleared = deploy(
        store, pd.read_parquet(args.bundle / "history.parquet"), manifest["version"]
    )
    logging.info(
        "seeded %d events in %.1fs (bundle %s, monitoring log %s)",
        n,
        time.perf_counter() - start,
        manifest["version"],
        "cleared" if cleared else "kept",
    )


if __name__ == "__main__":
    main()
