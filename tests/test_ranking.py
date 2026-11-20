"""Ranking stage: feature parity (no skew), stats by hand, candidates, LambdaRank, MMR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.data.schema import (
    AGE,
    GENDER,
    GENRES,
    IS_POSITIVE,
    ITEM_IDX,
    OCCUPATION,
    RATING,
    TIMESTAMP,
    USER_IDX,
    YEAR,
)
from recsys.eval.metrics import PAD, intra_list_diversity
from recsys.eval.protocol import EvalData, build_eval_data
from recsys.features.encoders import UserFeatures
from recsys.features.ranking import (
    SECONDS_PER_DAY,
    ItemTable,
    UserBlock,
    compute_features,
    feature_columns,
    feature_matrix,
    user_state,
    user_states_from_history,
)
from recsys.ranking.candidates import (
    label_candidates,
    merge_sources,
    retrieval_features,
    score_pairs,
    scores_to_recs,
    within_cap,
)
from recsys.ranking.mmr import mmr_rerank
from recsys.ranking.queries import build_queries
from recsys.ranking.ranker import LambdaRankRanker, RankingTable

DAY = SECONDS_PER_DAY


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            ITEM_IDX: [0, 1, 2, 3],
            GENRES: [["Drama"], ["Comedy"], ["Comedy", "Drama"], ["Horror"]],
            YEAR: pd.Series([1990, 2000, None, 1980], dtype="Int16"),
        }
    )


def _history() -> pd.DataFrame:
    rows = [  # (user, item, rating, day)
        (0, 0, 5, 1),
        (0, 1, 2, 3),
        (0, 2, 4, 9),
        (1, 0, 4, 2),
        (1, 3, 1, 10),
        (2, 2, 5, 5),
    ]
    df = pd.DataFrame(rows, columns=[USER_IDX, ITEM_IDX, RATING, "day"])
    df[TIMESTAMP] = (df.pop("day") * DAY).astype(np.int64)
    df[RATING] = df[RATING].astype(float)
    df[IS_POSITIVE] = df[RATING] >= 4
    return df


def _users() -> UserFeatures:
    return UserFeatures.from_users(
        pd.DataFrame(
            {
                USER_IDX: range(4),
                GENDER: ["F", "M", "F", "M"],
                AGE: [25, 35, 18, 25],
                OCCUPATION: [1, 2, 3, 4],
            }
        )
    )


# ------------------------------------------------------------------ item stats


def test_item_table_by_hand() -> None:
    table = ItemTable.from_history(_history(), _items())
    assert table.cutoff == 10 * DAY
    num = table.numeric
    # item 0: 2 ratings, 2 positives; global pos rate = 4/6, global mean rating = 21/6
    assert num.loc[0, "item_pos_rate"] == pytest.approx((2 + 10 * 4 / 6) / (2 + 10))
    assert num.loc[0, "item_mean_rating"] == pytest.approx((9 + 10 * 21 / 6) / (2 + 10))
    # no "age since first rating": on a dataset collected in one burst it is a calendar
    # clock (the drift monitor's day-0 finding), so it must not come back
    assert not any("age" in c for c in num.columns)
    assert not np.isnan(num.loc[3, "item_year"]) and np.isnan(num.loc[2, "item_year"])
    # 7-day window = days [3, 10]: positives there are item 2 (day 9) and item 2 (day 5)
    assert np.expm1(num["item_log_pop_7d"]).round().tolist() == [0, 0, 2, 0]
    assert table.genre_names == ["Comedy", "Drama", "Horror"]
    # positives: items 0, 2, 0, 2 -> Comedy 2/4, Drama 4/4, Horror 0
    np.testing.assert_allclose(table.global_genre_share, [0.5, 1.0, 0.0])


def test_user_state_by_hand_and_order_independent() -> None:
    table = ItemTable.from_history(_history(), _items())
    s = user_state([2, 0, 1], [4.0, 5.0, 2.0], [9 * DAY, 1 * DAY, 3 * DAY], [1, 1, 0], table)
    assert (s.n_ratings, s.n_pos) == (3.0, 2.0)
    assert s.mean_rating == pytest.approx(11 / 3)
    assert (s.days_since_last, s.span_days) == (1.0, 8.0)
    assert s.recent.tolist() == [0, 2]  # liked items, oldest first, whatever the input order
    assert s.mean_year == 1990.0  # item 2's year is unknown -> ignored
    # liked genres: Drama (item 0) + Comedy,Drama (item 2) -> counts [1, 2, 0], n = 2
    np.testing.assert_allclose(
        s.genre_share, (np.array([1, 2, 0]) + 5 * np.array([0.5, 1.0, 0.0])) / (2 + 5)
    )
    empty = user_state([], [], [], [], table)
    assert empty.n_ratings == 0 and np.isnan(empty.mean_rating) and empty.recent.size == 0


def test_offline_and_serving_feature_paths_are_identical() -> None:
    """Training-serving skew test: batch features == features built from one user's raw
    interaction list (what the API keeps in Redis), for every user and candidate."""
    history = _history()
    table = ItemTable.from_history(history, _items())
    vectors = np.random.default_rng(0).standard_normal((4, 3))
    users = np.array([0, 1, 2, 3])  # user 3 has no history (cold)
    rows = np.repeat(np.arange(4), 4)
    items = np.tile(np.arange(4), 4)
    retrieval = {"tt_score": np.linspace(0, 1, 16)}

    offline_block = UserBlock.stack(
        user_states_from_history(history, users, table), users, _users()
    )
    offline = compute_features(offline_block, rows, items, table, vectors, retrieval)

    for u in users:
        mine = history[history[USER_IDX] == u].sample(frac=1.0, random_state=int(u))  # any order
        state = user_state(mine[ITEM_IDX], mine[RATING], mine[TIMESTAMP], mine[IS_POSITIVE], table)
        block = UserBlock.stack([state], [u], _users())
        sel = rows == u
        online = compute_features(
            block,
            np.zeros(sel.sum(), np.int64),
            items[sel],
            table,
            vectors,
            {"tt_score": retrieval["tt_score"][sel]},
        )
        pd.testing.assert_frame_equal(online, offline[sel].reset_index(drop=True))


def test_cross_features_by_hand() -> None:
    history = _history()
    table = ItemTable.from_history(history, _items())
    vectors = np.eye(4)
    block = UserBlock.stack(user_states_from_history(history, [0], table), [0], _users())
    feats = compute_features(block, np.zeros(4, np.int64), np.arange(4), table, vectors, {})
    # user 0 liked items 0 and 2 (unit vectors): sim to item 2 is 1 (max), 0.5 mean, 1 last
    assert feats.loc[2, ["x_sim_recent_max", "x_sim_recent_mean", "x_sim_last"]].tolist() == [
        1.0,
        0.5,
        1.0,
    ]
    assert feats.loc[3, "x_sim_recent_max"] == 0.0
    assert feats.loc[1, "x_year_gap"] == 10.0  # item 1 (2000) vs mean liked year 1990


# ------------------------------------------------------------------ candidates


def test_merge_sources_union_ranks_and_labels() -> None:
    a = (np.array([[3, 1, PAD], [0, 2, 1]]), np.array([[0.9, 0.5, np.nan], [0.8, 0.7, 0.1]]))
    b = (np.array([[1, 4], [5, 0]]), np.array([[2.0, 1.0], [3.0, 2.5]]))
    cands = merge_sources({"a": a, "b": b})
    assert cands.rows.tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert cands.items.tolist() == [1, 3, 4, 0, 1, 2, 5]
    assert cands.retrieval["a_rank"].tolist() == [2, 1, 4, 1, 3, 2, 4]  # missing -> k + 1
    assert cands.retrieval["b_rank"].tolist() == [1, 3, 2, 2, 3, 3, 1]
    assert cands.retrieval["n_sources"].tolist() == [2, 1, 1, 2, 1, 1, 1]
    assert np.isnan(cands.retrieval["b_score"][1])

    data = EvalData(
        user_idx=np.array([10, 11]),
        relevant=[np.array([4, 9]), np.array([2])],
        seen=[np.empty(0, np.int64)] * 2,
        is_warm=np.array([True, True]),
        item_popularity=np.zeros(10, np.int64),
        history=pd.DataFrame(),
    )
    assert label_candidates(cands, data).tolist() == [0, 0, 1, 0, 0, 1, 0]


def test_scores_to_recs_orders_and_pads() -> None:
    rows = np.array([0, 0, 0, 2])
    items = np.array([5, 6, 7, 1])
    recs = scores_to_recs(rows, items, [0.1, 0.9, 0.5, 0.0], n_rows=3, k=2)
    assert recs.tolist() == [[6, 7], [PAD, PAD], [1, PAD]]


# ------------------------------------------------------------------ LambdaRank


def test_group_sizes_and_positive_filter() -> None:
    table = RankingTable(
        pd.DataFrame({"x": range(6)}), np.array([0, 1, 0, 0, 0, 1]), np.array([7, 7, 8, 8, 9, 9])
    )
    assert table.group_sizes().tolist() == [2, 2, 2]
    kept = table.with_positive_queries()
    assert kept.query.tolist() == [7, 7, 9, 9] and kept.features["x"].tolist() == [0, 1, 4, 5]
    with pytest.raises(ValueError, match="contiguous"):
        RankingTable(
            pd.DataFrame({"x": range(3)}), np.zeros(3, np.int64), np.array([1, 2, 1])
        ).group_sizes()


def test_lambdarank_learns_a_planted_signal() -> None:
    rng = np.random.default_rng(0)

    def make(n_queries: int, offset: int) -> RankingTable:
        n = n_queries * 20
        signal = rng.random(n)
        noise = rng.random(n)
        labels = (signal + 0.1 * rng.standard_normal(n) > 0.8).astype(np.int64)
        query = np.repeat(np.arange(offset, offset + n_queries), 20)
        return RankingTable(pd.DataFrame({"signal": signal, "noise": noise}), labels, query)

    train, valid = make(200, 0), make(50, 1000)
    ranker = LambdaRankRanker(
        params={"num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 5, "seed": 0},
        num_boost_round=200,
        early_stopping_rounds=20,
    ).fit(train, valid)
    importance = ranker.feature_importance()
    assert importance.loc[0, "feature"] == "signal" and importance.loc[0, "gain_share"] > 0.9
    # LambdaRank only rewards order near the top, so check what it optimizes: is each
    # query's top-scored item a positive? (Linear correlation with the raw signal is NOT a
    # fair test: scores of clear negatives are left flat, since their order costs nothing.)
    frame = pd.DataFrame(
        {"q": valid.query, "score": ranker.predict(valid.features), "label": valid.labels}
    )
    frame = frame[frame.groupby("q")["label"].transform("max") > 0]
    top1 = frame.loc[frame.groupby("q")["score"].idxmax(), "label"]
    assert top1.mean() > 0.95  # random would be ~0.2


# ------------------------------------------------------------------ MMR / diversity


def test_mmr_lambda_one_keeps_order_and_lambda_low_diversifies() -> None:
    # items 0 and 1 are near-duplicates; item 2 is different but slightly less relevant
    vectors = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]])
    cands = np.array([[0, 1, 2, PAD]])
    rel = np.array([[1.0, 0.95, 0.9, 0.0]])
    assert mmr_rerank(cands, rel, vectors, k=3, lam=1.0).tolist() == [[0, 1, 2]]
    assert mmr_rerank(cands, rel, vectors, k=2, lam=0.5).tolist() == [[0, 2]]
    with pytest.raises(ValueError):
        mmr_rerank(cands, rel, vectors, k=2, lam=1.5)


def test_intra_list_diversity_by_hand() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    ild = intra_list_diversity(np.array([[0, 1, 2], [0, 2, PAD], [1, PAD, PAD]]), vectors, k=3)
    # row 0: pair sims (0,1)=0, (0,2)=1, (1,2)=0 -> mean 1/3 -> ILD 2/3
    np.testing.assert_allclose(ild[:2], [2 / 3, 0.0])
    assert np.isnan(ild[2])


def test_score_pairs_reads_dense_scores_for_any_candidate() -> None:
    dense = np.arange(20, dtype=float).reshape(2, 10)  # user u, item i -> 10u + i
    data = EvalData(
        user_idx=np.array([0, 1]),
        relevant=[np.empty(0, np.int64)] * 2,
        seen=[np.empty(0, np.int64)] * 2,
        is_warm=np.array([True, True]),
        item_popularity=np.zeros(10, np.int64),
        history=pd.DataFrame(),
    )
    got = score_pairs(
        lambda u: dense[u], data, np.array([0, 1, 1]), np.array([3, 0, 9]), batch_size=1
    )
    assert got.tolist() == [3.0, 10.0, 19.0]


def test_ranker_windows_never_overlap_their_label_period() -> None:
    from types import SimpleNamespace

    from recsys.pipelines.ranker import make_windows

    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            USER_IDX: rng.integers(0, 30, n),
            ITEM_IDX: rng.integers(0, 15, n),
            RATING: rng.integers(1, 6, n).astype(float),
            TIMESTAMP: np.sort(rng.integers(0, 10_000, n)),
        }
    )
    df[IS_POSITIVE] = df[RATING] >= 4
    split = SimpleNamespace(
        train=df[df[TIMESTAMP] < 6000],
        val=df[(df[TIMESTAMP] >= 6000) & (df[TIMESTAMP] < 8000)],
        test=df[df[TIMESTAMP] >= 8000],
    )
    windows = make_windows(split, 0.85)
    cut = split.train[TIMESTAMP].quantile(0.85, interpolation="higher")
    assert windows["w1"].history[TIMESTAMP].max() < cut <= windows["w1"].target[TIMESTAMP].min()
    assert windows["w2"].history[TIMESTAMP].max() < 6000
    assert windows["w3"].history[TIMESTAMP].max() < 8000
    # no history row comes from the label period (w3 history is a re-indexed concat, so its
    # guard is the timestamp check above, not an index check)
    for name in ("w1", "w2"):
        w = windows[name]
        assert w.history.index.intersection(w.target.index).empty, name


def test_time_features_follow_the_decision_clock() -> None:
    """Offline the clock is the window cutoff; online it is request time. With a frozen
    cutoff, a user who interacts after the snapshot would get days_since_last < 0."""
    history = _history()
    table = ItemTable.from_history(history, _items())  # cutoff = day 10
    late = user_state([0, 1], [5.0, 4.0], [1 * DAY, 15 * DAY], [1, 1], table, as_of=20 * DAY)
    assert late.days_since_last == 5.0
    frozen = user_state([0, 1], [5.0, 4.0], [1 * DAY, 15 * DAY], [1, 1], table)
    assert frozen.days_since_last == -5.0  # the bug the explicit clock fixes


# ------------------------------------------------------------------ point-in-time queries


def _window() -> tuple[pd.DataFrame, pd.DataFrame]:
    """History = days 1-10 (`_history`), label period = days 11-20."""
    rows = [  # (user, item, rating, day); items 4+ are new to everyone
        (0, 3, 5, 12),  # user 0: like, dislike, like
        (0, 4, 1, 14),
        (0, 5, 4, 15),
        (1, 4, 5, 16),
        (1, 5, 2, 18),  # user 1's last event has no later like -> never a decision time
        (3, 2, 4, 11),  # user 3: no history at all (cold), one like
        (2, 6, 2, 13),  # user 2: no positive in the label period -> not a query user
    ]
    target = pd.DataFrame(rows, columns=[USER_IDX, ITEM_IDX, RATING, "day"])
    target[TIMESTAMP] = (target.pop("day") * DAY).astype(np.int64)
    target[RATING] = target[RATING].astype(float)
    target[IS_POSITIVE] = target[RATING] >= 4
    return _history(), target


def test_cutoff_queries_are_exactly_the_offline_protocol() -> None:
    history, target = _window()
    q = build_queries(history, target, 8, sessions_per_user=0, rng=np.random.default_rng(0))
    ref = build_eval_data(history, target, 8)
    assert q.user.tolist() == ref.user_idx.tolist() == [0, 1, 3]
    assert q.at_cutoff.all() and (q.as_of == 10 * DAY).all()
    for got, want in zip(q.data.relevant, ref.relevant, strict=True):
        assert sorted(got.tolist()) == sorted(want.tolist())
    for got, want in zip(q.data.seen, ref.seen, strict=True):
        assert sorted(got.tolist()) == sorted(want.tolist())
    assert q.data.is_warm.tolist() == ref.is_warm.tolist() == [True, True, False]


def test_session_queries_see_the_past_and_label_the_future() -> None:
    history, target = _window()
    q = build_queries(history, target, 8, sessions_per_user=10, rng=np.random.default_rng(0))
    sessions = np.flatnonzero(~q.at_cutoff)
    # user 0: decision times = days 12, 14, 15 (every event with a like at/after it);
    # user 1: day 16 only (day 18 has no later like); user 3: day 11
    got = {(int(q.user[i]), q.as_of[i] / DAY) for i in sessions}
    assert got == {(0, 12.0), (0, 14.0), (0, 15.0), (1, 16.0), (3, 11.0)}
    for i in sessions:
        h, t = q.history[i], q.as_of[i]
        assert (h.timestamps < t).all()  # nothing from the decision moment or later
        labels = q.data.relevant[i]
        assert len(labels) > 0 and not np.isin(labels, h.items).any()
        future = target[(target[USER_IDX] == q.user[i]) & (target[TIMESTAMP] >= t)]
        assert set(labels.tolist()) == set(future.loc[future[IS_POSITIVE], ITEM_IDX])
    # user 0 at day 15: history = 3 window rows + days 12 and 14 from the label period
    i = next(i for i in sessions if q.user[i] == 0 and q.as_of[i] == 15 * DAY)
    assert q.history[i].items.tolist() == [0, 1, 2, 3, 4]
    assert q.data.relevant[i].tolist() == [5]
    # a cold user becomes personalized once their label-period like is in the past
    cold = [i for i in range(q.n) if q.user[i] == 3]
    assert q.personalized[cold].tolist() == [False, False]  # day-11 like is the label itself
    with pytest.raises(ValueError, match="after the history cutoff"):
        build_queries(history, history, 8, 1, np.random.default_rng(0))


def test_retrieval_features_mark_fallback_scores_missing() -> None:
    """A retriever with no vector for a user scores with a fallback in another unit
    (BPR -> popularity counts); its score feature must be NaN, not a huge number."""
    dense = np.array([[1.0, 2.0, 3.0], [300.0, 200.0, 100.0]])  # row 1 = counts
    data = EvalData(
        user_idx=np.array([0, 1]),
        relevant=[np.empty(0, np.int64)] * 2,
        seen=[np.empty(0, np.int64)] * 2,
        is_warm=np.array([True, True]),
        item_popularity=np.zeros(3, np.int64),
        history=pd.DataFrame(),
    )
    recs = np.array([[2, 1], [0, 1]])
    cands = merge_sources({"bpr": (recs, np.take_along_axis(dense, recs, 1))})
    out = retrieval_features(cands, {"bpr": lambda u: dense[u]}, data, {"bpr": [True, False]})
    assert out["bpr_score"][cands.rows == 0].tolist() == [2.0, 3.0]
    assert np.isnan(out["bpr_score"][cands.rows == 1]).all()
    assert out["bpr_rank"][cands.rows == 1].tolist() == [1.0, 2.0]  # its ranking still counts


def test_serving_matrix_equals_the_training_frame() -> None:
    """The API skips the DataFrame (pandas overhead) and stacks the same columns straight
    into LightGBM's float matrix; values must be identical, in the ranker's order."""
    history = _history()
    table = ItemTable.from_history(history, _items())
    users = np.array([0, 1, 2])
    block = UserBlock.stack(user_states_from_history(history, users, table), users, _users())
    rows, items = np.array([0, 0, 1, 2]), np.array([3, 1, 2, 0])
    vecs = np.random.default_rng(0).normal(size=(4, 3))
    retrieval = {"tt_rank": np.array([1.0, 2, 1, 201]), "bpr_score": np.array([1.0, np.nan, 2, 3])}
    frame = compute_features(block, rows, items, table, vecs, retrieval)
    names = list(frame.columns[::-1])  # any order the ranker asks for
    matrix = feature_matrix(feature_columns(block, rows, items, table, vecs, retrieval), names)
    np.testing.assert_array_equal(matrix, frame[names].to_numpy(np.float64))


def test_score_cap_keeps_items_any_source_ranked_high() -> None:
    retrieval = {
        "tt_rank": np.array([1.0, 90, 201, 201]),
        "bpr_rank": np.array([201.0, 201, 5, 150]),
        "tt_score": np.array([0.9, 0.1, 0.2, 0.3]),  # scores never decide the cap
    }
    assert within_cap(retrieval, 80).tolist() == [True, False, True, False]
    assert within_cap(retrieval, None).all()
