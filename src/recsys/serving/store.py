"""Online feature store: each user's raw interaction log in Redis.

Key  recsys:u:<user_idx>   Redis list of "item_idx,rating,unix_ts" strings, append-only.

We store raw events, not precomputed features, on purpose: the API recomputes the user's
features with the same `user_state()` used to build training data, so there is one
feature implementation and no training-serving skew. Reading ~1k short strings per
request costs well under a millisecond.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import redis

from recsys.data.schema import ITEM_IDX, RATING, TIMESTAMP, USER_IDX

logger = logging.getLogger(__name__)

KEY_PREFIX = "recsys:u:"
VERSION_KEY = "recsys:bundle_version"
REC_STREAM = "recsys:log:recs"
FEATURE_STREAM = "recsys:log:features"


@dataclass(frozen=True)
class Interactions:
    items: npt.NDArray[np.int64]
    ratings: npt.NDArray[np.float64]
    timestamps: npt.NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.items)

    @classmethod
    def empty(cls) -> Interactions:
        return cls(np.empty(0, np.int64), np.empty(0, np.float64), np.empty(0, np.int64))

    def latest_per_item(self) -> Interactions:
        """Keep each item's most recent event (re-rating replaces the old rating), matching
        the offline `deduplicate_interactions` rule."""
        if len(self) == 0:
            return self
        order = np.lexsort((np.arange(len(self)), self.timestamps))  # time, then arrival
        items = self.items[order]
        _, last_from_end = np.unique(items[::-1], return_index=True)
        keep = np.sort(len(items) - 1 - last_from_end)
        sel = order[keep]
        return Interactions(self.items[sel], self.ratings[sel], self.timestamps[sel])


def _encode(item: int, rating: float, ts: int) -> str:
    return f"{item},{rating:g},{ts}"


class InteractionStore:
    def __init__(self, client: redis.Redis) -> None:
        self.client = client

    @classmethod
    def from_url(cls, url: str) -> InteractionStore:
        return cls(redis.Redis.from_url(url, decode_responses=True))

    def ping(self) -> bool:
        return bool(self.client.ping())

    def append(self, user_idx: int, item_idx: int, rating: float, timestamp: int) -> int:
        """Record one interaction; returns the user's new event count."""
        return int(
            self.client.rpush(f"{KEY_PREFIX}{user_idx}", _encode(item_idx, rating, timestamp))
        )

    def get(self, user_idx: int) -> Interactions:
        raw = self.client.lrange(f"{KEY_PREFIX}{user_idx}", 0, -1)
        if not raw:
            return Interactions.empty()
        parts = np.array([r.split(",") for r in raw])
        return Interactions(
            parts[:, 0].astype(np.int64),
            parts[:, 1].astype(np.float64),
            parts[:, 2].astype(np.int64),
        )

    def seed(self, history: pd.DataFrame, version: str, batch_users: int = 500) -> int:
        """Replace all users' logs with `history` (bulk load, pipelined). Returns #events."""
        for key in self.client.scan_iter(f"{KEY_PREFIX}*", count=1000):
            self.client.delete(key)
        ordered = history.sort_values([USER_IDX, TIMESTAMP, ITEM_IDX], kind="stable")
        pipe = self.client.pipeline(transaction=False)
        for n, (user, g) in enumerate(ordered.groupby(USER_IDX, sort=False), start=1):
            events = [
                _encode(int(i), float(r), int(t))
                for i, r, t in zip(g[ITEM_IDX], g[RATING], g[TIMESTAMP], strict=True)
            ]
            pipe.rpush(f"{KEY_PREFIX}{int(user)}", *events)
            if n % batch_users == 0:
                pipe.execute()
        pipe.set(VERSION_KEY, version)
        pipe.execute()
        logger.info("seeded %d events for %d users", len(ordered), ordered[USER_IDX].nunique())
        return len(ordered)

    def bundle_version(self) -> str | None:
        value = self.client.get(VERSION_KEY)
        return None if value is None else str(value)


class EventLog:
    """Monitoring log in two capped Redis streams, shared by all API workers.

    recsys:log:recs       one entry per /recommend: time, user, strategy, items, latency
    recsys:log:features   sampled candidate feature rows (for drift detection)
    """

    def __init__(self, client: redis.Redis, maxlen: int = 200_000) -> None:
        self.client = client
        self.maxlen = maxlen

    def log_recommendation(
        self,
        ts: float,
        user_idx: int | None,
        strategy: str,
        items: Sequence[int],
        n_candidates: int,
        latency_ms: float,
    ) -> None:
        self.client.xadd(
            REC_STREAM,
            {
                "ts": f"{ts:.0f}",
                "user": "" if user_idx is None else str(user_idx),
                "strategy": strategy,
                "items": ",".join(map(str, items)),
                "n_candidates": str(n_candidates),
                "latency_ms": f"{latency_ms:.3f}",
            },
            maxlen=self.maxlen,
            approximate=True,
        )

    def log_features(self, ts: float, rows: pd.DataFrame) -> None:
        self.client.xadd(
            FEATURE_STREAM,
            {"ts": f"{ts:.0f}", "rows": rows.to_json(orient="split", index=False)},
            maxlen=self.maxlen,
            approximate=True,
        )

    def recommendations(self, last: int | None = None) -> pd.DataFrame:
        frame = pd.DataFrame([fields for _, fields in self._read(REC_STREAM, last)])
        if frame.empty:
            return pd.DataFrame(
                columns=["ts", "user", "strategy", "items", "n_candidates", "latency_ms"]
            )
        frame["ts"] = frame["ts"].astype(np.int64)
        frame["user"] = pd.to_numeric(frame["user"], errors="coerce").astype("Int64")
        frame["items"] = frame["items"].map(lambda s: [int(x) for x in s.split(",")] if s else [])
        frame["n_candidates"] = frame["n_candidates"].astype(np.int64)
        frame["latency_ms"] = frame["latency_ms"].astype(np.float64)
        return frame

    def features(self, last: int | None = None) -> pd.DataFrame:
        """Logged candidate feature rows, oldest first, with each request's time as `_ts`.

        Entries are decoded with json.loads and stacked into one float matrix per column
        layout: one pd.read_json per entry cost ~9 ms, so /monitoring?last=5000 spent 45 s
        parsing. The values are the ranker's float matrix (JSON null = NaN)."""
        blocks: dict[tuple[str, ...], tuple[list[int], list[list[float | None]]]] = {}
        for _, fields in self._read(FEATURE_STREAM, last):
            payload = json.loads(fields["rows"])
            ts, data = blocks.setdefault(tuple(payload["columns"]), ([], []))
            ts.extend([int(fields["ts"])] * len(payload["data"]))
            data.extend(payload["data"])
        frames = []
        for columns, (ts, data) in blocks.items():
            frame = pd.DataFrame(
                np.array(data, dtype=np.float64).reshape(len(data), len(columns)),
                columns=list(columns),
            )
            frame.insert(0, "_ts", np.array(ts, dtype=np.int64))
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def clear(self) -> None:
        self.client.delete(REC_STREAM, FEATURE_STREAM)

    def _read(self, stream: str, last: int | None) -> list[tuple[str, dict[str, str]]]:
        if last is None:
            return list(self.client.xrange(stream))
        return list(reversed(self.client.xrevrange(stream, count=last)))
