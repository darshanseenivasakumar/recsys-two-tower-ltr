"""Temporal train / validation / test splits.

Two strategies:

global_temporal (default)
    Pick two timestamps t_val < t_test. Everything before t_val is train, [t_val, t_test)
    is validation, everything from t_test on is test. This simulates what happens in
    production: you train on all data up to a date, deploy, and are judged on what users
    do afterwards. Nothing in train happened after anything in test.

leave_last_n
    Each user's last n_test interactions are test and the n_val before those are
    validation. Common in academic papers. It is temporal *per user* but not *globally*:
    user A's test item may be from 2000 while user B's training items are from 2003, so
    the model trains on data from after some test events happened.

Why not random splits? A random split puts a user's 2003 ratings in train and their 2000
ratings in test, so the model learns from the future to predict the past. It can also see
a movie's later popularity, and sequence models can read the answer from the history
that comes after it. Offline numbers come out inflated and don't hold up online.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from recsys.config import SplitConfig
from recsys.data.schema import IS_POSITIVE, ITEM_ID, TIMESTAMP, USER_ID

logger = logging.getLogger(__name__)

PARTS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    def parts(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "test": self.test}


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def _split_from_labels(
    interactions: pd.DataFrame, labels: pd.Series, meta: dict[str, Any]
) -> SplitResult:
    """Build a SplitResult from a per-row Series of 'train' / 'val' / 'test' labels."""
    unknown = set(labels.unique()) - set(PARTS)
    if unknown:
        raise ValueError(f"Unexpected split labels: {unknown}")
    frames = {p: interactions[labels == p].reset_index(drop=True) for p in PARTS}
    return SplitResult(train=frames["train"], val=frames["val"], test=frames["test"], meta=meta)


def global_temporal_split(
    interactions: pd.DataFrame, val_frac: float, test_frac: float
) -> SplitResult:
    """Split at two global timestamps chosen as quantiles of the interaction times.

    E.g. val_frac=test_frac=0.1 puts the cutoffs at the 80th and 90th percentile of all
    timestamps. Comparisons are strict (`< t_val`), so interactions sharing a timestamp
    with a cutoff all land on the same side, never split across train and test.
    """
    ts = interactions[TIMESTAMP]
    t_val = int(ts.quantile(1.0 - val_frac - test_frac, interpolation="higher"))
    t_test = int(ts.quantile(1.0 - test_frac, interpolation="higher"))

    labels = pd.Series("train", index=interactions.index)
    labels[ts >= t_val] = "val"
    labels[ts >= t_test] = "test"

    meta = {
        "strategy": "global_temporal",
        "t_val": t_val,
        "t_test": t_test,
        "t_val_iso": _iso(t_val),
        "t_test_iso": _iso(t_test),
    }
    return _split_from_labels(interactions, labels, meta)


def leave_last_n_split(interactions: pd.DataFrame, n_val: int, n_test: int) -> SplitResult:
    """Hold out each user's last `n_test` interactions as test and the `n_val` before as val.

    "Last" means latest timestamp; ties are broken by item_id so the split is
    deterministic regardless of input row order.

    Contract (enforced by tests/test_split.py):
      * train, val and test partition the input (disjoint; together they cover every row).
      * For users with more than n_val + n_test interactions, test holds exactly their
        last n_test and val exactly the n_val before those.
      * Every user keeps at least one training interaction, so no user is "cold" at
        evaluation time purely because of how we split.
    """
    # Sort so that, within each user, rows are oldest -> newest with a deterministic tie-break.
    ordered = interactions.sort_values([USER_ID, TIMESTAMP, ITEM_ID], kind="stable")

    by_user = ordered.groupby(USER_ID)
    pos_from_end = by_user.cumcount(ascending=False)  # 0 = user's newest interaction
    history_len = by_user[ITEM_ID].transform("size")

    # Policy for short histories: a user with <= n_val + n_test interactions keeps
    # everything in train. Holding out a partial test instead would leave them with
    # (almost) no history, so a "cold" user would be produced by the split itself
    # rather than by the data.
    eligible = history_len > n_val + n_test

    labels = pd.Series("train", index=ordered.index)
    labels[eligible & (pos_from_end < n_test)] = "test"
    labels[eligible & (pos_from_end >= n_test) & (pos_from_end < n_test + n_val)] = "val"

    meta = {
        "strategy": "leave_last_n",
        "n_val": n_val,
        "n_test": n_test,
        "users_too_short_to_hold_out": int(ordered.loc[~eligible, USER_ID].nunique()),
    }
    return _split_from_labels(ordered, labels, meta)


def make_split(interactions: pd.DataFrame, cfg: SplitConfig) -> SplitResult:
    if cfg.strategy == "global_temporal":
        return global_temporal_split(interactions, cfg.val_frac, cfg.test_frac)
    return leave_last_n_split(interactions, cfg.n_val, cfg.n_test)


def summarize_split(split: SplitResult) -> dict[str, Any]:
    """Statistics that reveal whether a split is sensible (and how 'cold' val/test are)."""
    train_users = set(split.train[USER_ID].unique())
    train_items = set(split.train[ITEM_ID].unique())
    summary: dict[str, Any] = {"meta": split.meta}

    for name, df in split.parts().items():
        positives = df[df[IS_POSITIVE]]
        stats: dict[str, Any] = {
            "interactions": len(df),
            "positives": len(positives),
            "users": int(df[USER_ID].nunique()),
            "items": int(df[ITEM_ID].nunique()),
        }
        if len(df):
            stats["time_range"] = [_iso(int(df[TIMESTAMP].min())), _iso(int(df[TIMESTAMP].max()))]
        if name != "train" and len(df):
            pos_users = set(positives[USER_ID].unique())
            stats["users_with_positive"] = len(pos_users)
            # Warm users have training history, so non-content models can score them.
            stats["warm_users_with_positive"] = len(pos_users & train_users)
            stats["cold_users"] = len(set(df[USER_ID].unique()) - train_users)
            stats["cold_items"] = len(set(df[ITEM_ID].unique()) - train_items)
            stats["positives_on_cold_items"] = int((~positives[ITEM_ID].isin(train_items)).sum())
        summary[name] = stats
    return summary


def save_split(split: SplitResult, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in split.parts().items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    summary = summarize_split(split)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote split to %s", out_dir)
    return summary


def load_split(in_dir: Path) -> SplitResult:
    frames = {p: pd.read_parquet(in_dir / f"{p}.parquet") for p in PARTS}
    meta = json.loads((in_dir / "summary.json").read_text(encoding="utf-8"))["meta"]
    return SplitResult(train=frames["train"], val=frames["val"], test=frames["test"], meta=meta)
