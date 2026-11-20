"""Serving: Redis store, offline/online feature parity on real users, HTTP API.

The parity and API tests use the real ML-1M serving bundle and are skipped if it hasn't
been built (python -m recsys.serving.bundle ...). Redis is fakeredis: same client API,
same code path, no server needed.
"""

from __future__ import annotations

from pathlib import Path

import fakeredis
import numpy as np
import pandas as pd
import pytest

from recsys.data.schema import ITEM_IDX, RATING, TIMESTAMP, USER_IDX
from recsys.ranking.candidates import top_scored, within_cap
from recsys.serving.store import Interactions, InteractionStore

BUNDLE = Path("artifacts/serving/ml-1m")
W3 = Path("artifacts/ranker/ml-1m/windows/w3")
needs_bundle = pytest.mark.skipif(
    not (BUNDLE / "manifest.json").exists() or not (W3 / "pairs.parquet").exists(),
    reason="serving bundle / ranker windows not built",
)


def _store() -> InteractionStore:
    return InteractionStore(fakeredis.FakeRedis(decode_responses=True))


# ------------------------------------------------------------------ store


def test_store_roundtrip_and_seed_replaces_old_data() -> None:
    store = _store()
    store.append(5, 1, 4.0, 100)
    history = pd.DataFrame(
        {USER_IDX: [3, 3, 7], ITEM_IDX: [9, 2, 4], RATING: [5.0, 3.0, 4.0], TIMESTAMP: [20, 10, 5]}
    )
    assert store.seed(history, "v1") == 3
    assert len(store.get(5)) == 0  # seeding wipes previous logs
    got = store.get(3)
    assert got.items.tolist() == [2, 9] and got.timestamps.tolist() == [10, 20]  # time order
    assert store.bundle_version() == "v1"
    assert store.append(3, 4, 2.5, 30) == 3
    assert store.get(3).ratings.tolist() == [3.0, 5.0, 2.5]


def test_rerating_keeps_the_latest_event_per_item() -> None:
    events = Interactions(
        np.array([1, 2, 1, 3]), np.array([2.0, 5.0, 4.0, 1.0]), np.array([10, 11, 12, 12])
    )
    latest = events.latest_per_item()
    assert sorted(zip(latest.items.tolist(), latest.ratings.tolist(), strict=True)) == [
        (1, 4.0),
        (2, 5.0),
        (3, 1.0),
    ]


# ------------------------------------------------------------------ parity (real data)


@pytest.fixture(scope="module")
def real() -> dict:
    if not (BUNDLE / "manifest.json").exists() or not (W3 / "pairs.parquet").exists():
        pytest.skip("serving bundle / ranker windows not built")
    from recsys.config import load_data_config
    from recsys.data.split import load_split
    from recsys.eval.protocol import eval_data_for_stage
    from recsys.pipelines.ranker import WindowTable
    from recsys.serving.bundle import ServingBundle
    from recsys.serving.service import RecommenderService

    bundle = ServingBundle.load(BUNDLE)
    split = load_split(load_data_config("configs/data/ml-1m.yaml").split_path)
    test_data = eval_data_for_stage(split, "test", bundle.n_items)
    rows = np.random.default_rng(0).choice(test_data.n_users, size=25, replace=False)
    users = test_data.user_idx[rows]
    history = pd.read_parquet(BUNDLE / "history.parquet")
    store = _store()
    store.seed(history[history[USER_IDX].isin(users)], bundle.manifest["version"])
    return {
        "service": RecommenderService(bundle, store),
        "window": WindowTable.load(W3),
        "rows": rows,
        "users": users,
        "store": store,
    }


@needs_bundle
def test_serving_features_equal_the_offline_ranker_features(real: dict) -> None:
    """The training-serving skew test on real users: the API's candidate set and feature
    rows must equal what the ranker was evaluated on (Phase 4, test window)."""
    svc, window = real["service"], real["window"]
    flipped, cells = 0, 0
    for row, user in zip(real["rows"], real["users"], strict=True):
        online = svc.pair_features(int(user), real["store"].get(int(user)).latest_per_item())
        # serving scores only candidates within the bundle's score cap (some retriever
        # ranked them in its top `cap`); the same rule applied to the offline rows
        sel = (window.rows == row) & within_cap(window.features, svc.score_cap)
        np.testing.assert_array_equal(online.candidates.items, window.items[sel])
        offline = window.features.loc[sel, svc.feature_names].reset_index(drop=True)
        online_f = online.features.reset_index(drop=True)
        # Retriever scores come from float32 matmuls: offline scored 1024 users per GEMM,
        # serving scores one user, and BLAS sums in a different order for different
        # shapes -> last-bit differences (~1e-6 absolute). Allow exactly that, nothing more.
        # The embedding similarities are float32 GEMMs too: offline over all ~270 candidates,
        # online over the ~110 within the score cap -> same last-bit effect (~2e-7).
        float32_scores = [
            "tt_score",
            "bpr_score",
            "x_sim_recent_max",
            "x_sim_recent_mean",
            "x_sim_last",
        ]
        for col in float32_scores:
            np.testing.assert_allclose(online_f[col], offline[col], rtol=0, atol=1e-5)
            assert (online_f[col].isna() == offline[col].isna()).all()
        # ...which can swap two candidates whose scores tie to within that noise. Measured
        # on 200 users: 8 of 108k rank cells differ, every one by exactly 1.
        rank_cols = ["tt_rank", "bpr_rank"]
        delta = np.abs(online_f[rank_cols].to_numpy() - offline[rank_cols].to_numpy())
        assert delta.max() <= 1  # only neighbours swap
        flipped, cells = flipped + int((delta > 0).sum()), cells + delta.size
        exact = [c for c in svc.feature_names if c not in float32_scores + rank_cols]
        # values, not dtypes: the API hands LightGBM one float64 matrix (demographic codes
        # included), the offline table keeps pandas' int columns
        pd.testing.assert_frame_equal(
            online_f[exact], offline[exact], check_exact=False, rtol=1e-9, check_dtype=False
        )

        # what the user sees: the ranker's top-10 must be identical
        items = online.candidates.items
        zeros = np.zeros(len(items), dtype=np.int64)
        on_top = top_scored(zeros, items, svc.booster.predict(online_f), 1, 10)[0]
        off_top = top_scored(zeros, items, svc.booster.predict(offline), 1, 10)[0]
        np.testing.assert_array_equal(on_top, off_top)
    assert flipped / cells < 1e-3  # aggregate: 8 of 108k cells on 200 users


# ------------------------------------------------------------------ HTTP API


@pytest.fixture(scope="module")
def client(real: dict):  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from recsys.serving.api import create_app

    with TestClient(create_app(real["service"])) as c:
        yield c


@needs_bundle
def test_recommend_warm_user_excludes_seen_items(real: dict, client) -> None:  # type: ignore[no-untyped-def]
    svc = real["service"]
    user_idx = int(real["users"][0])
    user_id = next(uid for uid, idx in svc.bundle.user_index.items() if idx == user_idx)
    body = client.get("/recommend", params={"user_id": user_id, "k": 10}).json()
    assert body["strategy"] == "personalized" and len(body["items"]) == 10
    seen = {
        svc.bundle.catalog.item_id.astype(str).iloc[i] for i in real["store"].get(user_idx).items
    }
    assert not seen & {it["item_id"] for it in body["items"]}
    assert body["timings_ms"]["total"] > 0


@needs_bundle
def test_cold_start_then_first_like_switches_to_personalized(real: dict, client) -> None:  # type: ignore[no-untyped-def]
    svc = real["service"]
    unseeded = next(
        (uid, idx) for uid, idx in svc.bundle.user_index.items() if idx not in set(real["users"])
    )
    user_id, user_idx = unseeded
    assert len(real["store"].get(user_idx)) == 0  # not seeded in this fake Redis

    cold = client.get("/recommend", params={"user_id": user_id}).json()
    assert cold["strategy"] == "popular" and len(cold["items"]) == 10

    first = cold["items"][0]["item_id"]
    resp = client.post("/interaction", json={"user_id": user_id, "item_id": first, "rating": 5})
    assert resp.status_code == 202 and resp.json()["n_interactions"] == 1
    warm = client.get("/recommend", params={"user_id": user_id}).json()
    assert warm["strategy"] == "personalized"
    assert first not in {it["item_id"] for it in warm["items"]}  # just-rated item excluded

    unknown = client.get("/recommend", params={"user_id": "no-such-user"}).json()
    assert unknown["strategy"] == "popular"


@needs_bundle
def test_api_validation(client) -> None:  # type: ignore[no-untyped-def]
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/recommend", params={"user_id": "1", "k": 0}).status_code == 422
    bad_item = {"user_id": "1", "item_id": "999999", "rating": 4}
    assert client.post("/interaction", json=bad_item).status_code == 404
    bad_rating = {"user_id": "1", "item_id": "1", "rating": 9}
    assert client.post("/interaction", json=bad_rating).status_code == 422


@needs_bundle
def test_clock_moves_time_features_only(real: dict) -> None:
    """A later decision time shifts days_since_last by exactly the elapsed days and leaves
    every other feature unchanged (no feature may be a calendar clock)."""
    svc = real["service"]
    user = int(real["users"][0])
    events = real["store"].get(user).latest_per_item()
    cutoff = float(svc.bundle.manifest["cutoff_unix"])
    at_cutoff = svc.pair_features(user, events, as_of=cutoff).features
    later = svc.pair_features(user, events, as_of=cutoff + 30 * 86_400).features
    diff = later["user_days_since_last"] - at_cutoff["user_days_since_last"]
    np.testing.assert_allclose(diff, 30.0, atol=1e-9)
    others = [c for c in svc.feature_names if c != "user_days_since_last"]
    pd.testing.assert_frame_equal(later[others], at_cutoff[others])


@needs_bundle
def test_user_unknown_to_bpr_gets_missing_bpr_score(real: dict) -> None:
    """BPR's factors freeze at bundle time. A user whose first like comes later is
    retrieved by BPR's popularity fallback, but bpr_score must be NaN, as in training."""
    svc = real["service"]
    cold = int(np.flatnonzero(~svc.bundle.bpr.warm)[0])
    first_item = int(svc.bundle.popular[0])
    events = Interactions(np.array([first_item]), np.array([5.0]), np.array([0]))
    feats = svc.pair_features(cold, events).features
    assert feats["bpr_score"].isna().all() and feats["tt_score"].notna().all()
    assert (feats["bpr_rank"] <= svc.k["bpr"]).any()  # BPR still contributes candidates


@needs_bundle
def test_compiled_ranker_serves_exactly_the_same_recommendations(
    real: dict, request: pytest.FixtureRequest
) -> None:
    """The lleaves backend on the real bundle: same items, same scores, for every test user.
    Compiling the 292 trees takes ~3 min once; the binary is kept in pytest's cache."""
    pytest.importorskip("lleaves")
    from recsys.serving.service import RecommenderService

    base = real["service"]
    cache = Path(request.config.cache.mkdir("lleaves-ranker"))
    fast = RecommenderService(
        base.bundle, real["store"], ranker_backend="lleaves", ranker_cache_dir=cache
    )
    assert fast.ranker_backend == "lleaves"  # verification passed, no silent fallback
    ids = {idx: uid for uid, idx in base.bundle.user_index.items()}
    for user in real["users"]:
        a = base.recommend(ids[int(user)], 10)
        b = fast.recommend(ids[int(user)], 10)
        assert a.item_idx == b.item_idx
        assert [it["score"] for it in a.items] == [it["score"] for it in b.items]


@needs_bundle
def test_monitoring_endpoint_reports_traffic_and_drift(real: dict) -> None:
    from fastapi.testclient import TestClient

    from recsys.monitoring.reference import REFERENCE_FILE
    from recsys.serving.api import create_app
    from recsys.serving.service import RecommenderService
    from recsys.serving.store import EventLog

    base = real["service"]
    log = EventLog(real["store"].client)
    log.clear()
    svc = RecommenderService(
        base.bundle, real["store"], event_log=log, feature_sample_rate=1.0, seed=0
    )
    reference = pd.read_parquet(BUNDLE / REFERENCE_FILE)
    ids = {idx: uid for uid, idx in base.bundle.user_index.items()}
    with TestClient(create_app(svc, reference)) as c:
        for user in real["users"]:
            r = c.get("/recommend", params={"user_id": ids[int(user)]})
            assert r.headers["X-Request-ID"]
        c.get("/recommend", params={"user_id": "no-such-user"})
        body = c.get("/monitoring").json()
    # users with no positive rating in train+val get the cold-start path (2 of these 25)
    warm = sum(
        bool((real["store"].get(int(u)).ratings >= svc.threshold).any()) for u in real["users"]
    )
    n = len(real["users"]) + 1
    assert body["n_requests"] == n
    assert body["cold_start_share"] == pytest.approx((n - warm) / n)
    assert 0 < body["exposure"]["catalog_coverage"] < 1
    drift = body["drift"]  # each personalized request logs 10 sampled candidate rows
    assert drift["n_rows"] == 10 * warm and drift["max_psi"] >= 0 and len(drift["top"]) == 5
    log.clear()
