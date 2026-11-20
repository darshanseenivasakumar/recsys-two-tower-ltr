"""Monitoring: PSI by hand, exposure stats, event log round-trip, replay scoring, JSON logs."""

from __future__ import annotations

import json
import logging
import math

import fakeredis
import numpy as np
import pandas as pd
import pytest

from recsys.data.eda import gini
from recsys.data.schema import ITEM_IDX, TIMESTAMP, USER_IDX
from recsys.monitoring.drift import EPS, Binning, drift_table, psi, status
from recsys.monitoring.exposure import exposure_summary
from recsys.monitoring.reference import sample_per_query
from recsys.monitoring.simulate import future_hits
from recsys.serving.logs import JsonFormatter, request_id_var
from recsys.serving.seed import deploy
from recsys.serving.store import EventLog, InteractionStore

# ------------------------------------------------------------------ PSI


def test_psi_by_hand() -> None:
    reference = np.arange(100, dtype=float)  # quartile edges 24.75, 49.5, 74.25
    current = np.zeros(50)  # everything lands in the first bin
    ref = np.array([0.25, 0.25, 0.25, 0.25, EPS])  # 4 value bins + missing bin
    cur = np.array([1.0, EPS, EPS, EPS, EPS])
    expected = float(np.sum((cur - ref) * np.log(cur / ref)))
    assert psi(reference, current, bins=4) == pytest.approx(expected)
    assert psi(reference, reference, bins=4) == pytest.approx(0.0)


def test_psi_flags_new_missingness_and_unseen_categories() -> None:
    flags = np.array([0.0, 1.0] * 50)  # 2 values -> treated as categorical
    assert Binning.fit(flags).categories is not None
    assert psi(flags, flags) == pytest.approx(0.0)
    unseen = np.array([2.0] * 100)  # a category training never saw: its own bin
    assert psi(flags, unseen) > 1.0
    numeric = np.random.default_rng(0).normal(size=1000)
    half_missing = numeric.copy()
    half_missing[::2] = np.nan
    assert psi(numeric, half_missing) > 1.0  # NaN has its own bin: missingness is drift


def test_drift_table_orders_by_psi_and_labels_status() -> None:
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"same": rng.normal(size=2000), "shifted": rng.normal(size=2000)})
    cur = pd.DataFrame({"same": rng.normal(size=2000), "shifted": rng.normal(1.5, 1, 2000)})
    table = drift_table(ref, cur, ["same", "shifted"])
    assert table["feature"].tolist() == ["shifted", "same"]
    assert table.loc[0, "status"] == "major" and table.loc[1, "status"] == "stable"
    assert (status(0.05), status(0.15), status(0.3)) == ("stable", "moderate", "major")


# ------------------------------------------------------------------ exposure


def test_exposure_summary_by_hand() -> None:
    popularity = [10, 5, 1, 0]  # head (top 25%) = item 0
    s = exposure_summary([[0, 1], [0, 2]], popularity, head_frac=0.25)
    assert s["n_impressions"] == 4 and s["catalog_coverage"] == 0.75
    assert s["avg_popularity"] == pytest.approx((10 + 5 + 10 + 1) / 4)
    assert s["long_tail_share"] == pytest.approx(0.5)
    assert s["exposure_gini"] == pytest.approx(gini([2, 1, 1, 0]))


# ------------------------------------------------------------------ event log


def test_event_log_round_trip_keeps_nan() -> None:
    log = EventLog(fakeredis.FakeRedis(decode_responses=True))
    log.log_recommendation(1000.0, 7, "personalized", [3, 1, 2], 250, 12.5)
    log.log_recommendation(1001.0, None, "popular", [5], 0, 0.4)
    log.log_features(1000.0, pd.DataFrame({"a": [1.0, np.nan], "b": [0.5, 2.0]}))
    recs = log.recommendations()
    assert recs["strategy"].tolist() == ["personalized", "popular"]
    assert recs["items"].tolist() == [[3, 1, 2], [5]]
    assert recs["user"].isna().tolist() == [False, True]
    assert log.recommendations(last=1)["ts"].tolist() == [1001]
    feats = log.features()
    assert feats["_ts"].tolist() == [1000, 1000] and math.isnan(feats.loc[1, "a"])
    log.log_features(1002.0, pd.DataFrame({"a": [0.125], "b": [-3.5]}))
    feats = log.features()  # several entries: oldest first, one row per logged row
    assert feats["_ts"].tolist() == [1000, 1000, 1002]
    assert feats["a"].dtype == np.float64 and feats.loc[2, ["a", "b"]].tolist() == [0.125, -3.5]
    assert log.features(last=1)["_ts"].tolist() == [1002]  # last = most recent entries
    log.clear()
    assert log.recommendations().empty and log.features().empty


def test_new_bundle_clears_the_monitoring_log_same_bundle_keeps_it() -> None:
    client = fakeredis.FakeRedis(decode_responses=True)
    store, log = InteractionStore(client), EventLog(client)
    history = pd.DataFrame({USER_IDX: [1], ITEM_IDX: [2], "rating": [5.0], TIMESTAMP: [10]})
    assert deploy(store, history, "v1") == (1, True)
    log.log_recommendation(11.0, 1, "personalized", [3], 200, 9.0)
    assert deploy(store, history, "v1") == (1, False)  # restart: same model, keep its log
    assert len(log.recommendations()) == 1
    assert deploy(store, history, "v2") == (1, True)  # new model: old rows don't belong
    assert log.recommendations().empty


def test_reference_sampling_matches_serving_sampling() -> None:
    query = np.repeat([0, 1, 2], [30, 5, 12])
    picked = sample_per_query(query, 10, np.random.default_rng(0))
    assert np.bincount(query[picked]).tolist() == [10, 5, 10]  # up to 10 per query
    assert len(set(picked.tolist())) == len(picked)


# ------------------------------------------------------------------ replay scoring


def test_future_hits_uses_only_the_next_horizon() -> None:
    day = 86_400
    positives = pd.DataFrame(
        {USER_IDX: [1, 1, 1], ITEM_IDX: [10, 11, 12], TIMESTAMP: [day, 3 * day, 20 * day]}
    )
    requests = pd.DataFrame(
        {
            "user": [1, 1, 2],
            "ts": [day, 5 * day, day],
            "served": [[11, 99], [12], [10]],
            "popular": [[99], [11], [10]],
        }
    )
    out = future_hits(requests, positives, horizon_days=7)
    # request 0 (day 1): likes 10, 11 in [1, 8] -> served hits (11), popular misses
    # request 1 (day 5): nothing in [5, 12]; item 12 is on day 20, item 11 already past
    # request 2: user 2 has no likes
    assert out["n_future_likes"].tolist() == [2, 0, 0]
    assert out["hit_served"].tolist() == [True, False, False]
    assert out["hit_popular"].tolist() == [False, False, False]


# ------------------------------------------------------------------ logging


def test_json_log_lines_carry_request_id_and_extras() -> None:
    record = logging.LogRecord("recsys.api", logging.INFO, "f", 1, "recommend", (), None)
    record.strategy = "popular"  # what logger.info(..., extra={...}) attaches
    token = request_id_var.set("abc123")
    try:
        line = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert line["msg"] == "recommend" and line["request_id"] == "abc123"
    assert line["strategy"] == "popular" and line["level"] == "INFO"
    assert line["pid"] == record.process  # per-worker load is read from this
