"""Baseline models: hand-computed similarities and planted-structure recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys.data.schema import IS_POSITIVE, ITEM_IDX, TIMESTAMP, USER_IDX
from recsys.models import build_model
from recsys.models.base import ColdStartFallback
from recsys.models.bpr import sample_negatives
from recsys.models.item_knn import ItemKNN, topk_cosine_similarity
from recsys.models.matrix import interaction_matrix
from recsys.models.popular import MostPopular
from recsys.pipelines.baselines import expand_grid, merge_rows, mlflow_key


def _history(rows: list[tuple[int, int, bool, int]]) -> pd.DataFrame:
    """rows: (user_idx, item_idx, is_positive, timestamp)."""
    return pd.DataFrame(rows, columns=[USER_IDX, ITEM_IDX, IS_POSITIVE, TIMESTAMP])


# Users x items for the cosine examples:
#   u0 likes {0, 1}, u1 likes {0, 1, 2}, u2 likes {2}
#   item user-sets: 0 -> {0,1}, 1 -> {0,1}, 2 -> {1,2}; every norm = sqrt(2)
#   cos(0,1) = 2/2 = 1.0,  cos(0,2) = 1/2 = 0.5,  cos(1,2) = 1/2 = 0.5
KNN_HISTORY = _history(
    [
        (0, 0, True, 1),
        (0, 1, True, 1),
        (1, 0, True, 1),
        (1, 1, True, 1),
        (1, 2, True, 1),
        (2, 2, True, 1),
    ]
)


def test_interaction_matrix_is_binary_positives_with_catalog_shape() -> None:
    hist = _history([(0, 1, True, 1), (0, 2, False, 2), (1, 1, True, 3)])
    m = interaction_matrix(hist, n_users=3, n_items=4)
    assert m.shape == (3, 4)
    assert m.toarray().tolist() == [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]


def test_cosine_similarity_hand_computed() -> None:
    x = interaction_matrix(KNN_HISTORY, 3, 3)
    sim = topk_cosine_similarity(x, k=2).toarray()
    np.testing.assert_allclose(sim, [[0, 1.0, 0.5], [1.0, 0, 0.5], [0.5, 0.5, 0]], atol=1e-6)


def test_shrinkage_and_topk_pruning() -> None:
    x = interaction_matrix(KNN_HISTORY, 3, 3)
    # shrink=2: cos(0,1) = 2/(2+2) = 0.5, cos(0,2) = 1/(2+2) = 0.25
    shrunk = topk_cosine_similarity(x, k=2, shrink=2.0).toarray()
    assert shrunk[0, 1] == pytest.approx(0.5, abs=1e-6)
    assert shrunk[0, 2] == pytest.approx(0.25, abs=1e-6)
    # k=1 keeps only each item's single nearest neighbour (rows 0 and 1 have no tie).
    pruned = topk_cosine_similarity(x, k=1)
    assert pruned.getnnz(axis=1)[:2].tolist() == [1, 1]
    assert pruned[0, 1] == pytest.approx(1.0)


def test_blocked_similarity_matches_unblocked() -> None:
    rng = np.random.default_rng(0)
    x = sp.csr_matrix((rng.random((50, 30)) < 0.2).astype(np.float32))
    a = topk_cosine_similarity(x, k=30, block_size=7).toarray()
    b = topk_cosine_similarity(x, k=30, block_size=1024).toarray()
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_item_knn_scores_are_sum_of_neighbour_similarities() -> None:
    model = ItemKNN(k_neighbors=2).fit(KNN_HISTORY, 3, 3)
    # u2 likes only item 2 -> scores = row 2 of the similarity matrix.
    np.testing.assert_allclose(model.score(np.array([2])), [[0.5, 0.5, 0.0]], atol=1e-6)


def test_most_popular_counts_positives_and_respects_window() -> None:
    day = 86_400
    hist = _history(
        [(0, 0, True, 0), (1, 0, True, 1), (2, 1, True, 10 * day), (0, 2, False, 10 * day)]
    )
    all_time = MostPopular().fit(hist, 3, 3).score(np.array([0, 1]))
    assert all_time.argmax(axis=1).tolist() == [0, 0]  # item 0 has 2 positives
    recent = MostPopular(window_days=2).fit(hist, 3, 3).score(np.array([0]))
    assert recent.argmax() == 1  # only item 1 was liked within the last 2 days
    assert recent[0, 2] < recent[0, 0]  # negatives never count as popularity


def test_cold_start_fallback_routes_users_without_positives() -> None:
    hist = _history([(0, 1, True, 1), (1, 1, True, 2), (1, 0, True, 3), (2, 2, False, 4)])
    model = ColdStartFallback(ItemKNN(k_neighbors=5), MostPopular()).fit(hist, 4, 3)
    scores = model.score(np.array([0, 2, 3]))
    popular = MostPopular().fit(hist, 4, 3).score(np.array([0]))[0]
    np.testing.assert_allclose(scores[1], popular)  # user 2: only a negative -> fallback
    np.testing.assert_allclose(scores[2], popular)  # user 3: no history -> fallback
    assert model.params() == {"k_neighbors": 5, "shrink": 0.0}


def test_sample_negatives_avoids_positives() -> None:
    rng = np.random.default_rng(0)
    n_items = 10
    users = np.repeat(np.arange(5), 200)
    liked = {u: set(range(9)) for u in range(5)}  # every user likes items 0..8 -> only 9 is free
    keys = np.sort([u * n_items + i for u, items in liked.items() for i in items])
    neg = sample_negatives(users, keys, n_items, rng, max_tries=50)
    assert (neg == 9).mean() > 0.99


def _planted_history() -> tuple[pd.DataFrame, list[set[int]]]:
    """Users 0-19 like items 0-4, users 20-39 like items 5-9; each user is missing one."""
    rows, groups = [], []
    for u in range(40):
        group = set(range(5)) if u < 20 else set(range(5, 10))
        held_out = min(group) + u % 5
        rows += [(u, i, True, 1) for i in group - {held_out}]
        groups.append(group)
    return _history(rows), groups


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("item_knn", {"k_neighbors": 10}),
        ("als", {"factors": 4, "iterations": 15, "regularization": 0.01, "alpha": 10.0}),
        # reg matters: with reg=1e-5 BPR memorizes that each user's held-out item is a
        # sampled "negative" (loss -> 0.001, hit rate falls as epochs grow).
        ("bpr", {"dim": 8, "epochs": 60, "lr": 0.05, "batch_size": 32, "reg": 1.0e-2}),
    ],
)
def test_models_recover_planted_groups(name: str, params: dict[str, float]) -> None:
    hist, groups = _planted_history()
    model = build_model(name, params).fit(hist, 40, 10)
    scores = model.score(np.arange(40)).copy()
    seen = interaction_matrix(hist, 40, 10).toarray().astype(bool)
    scores[seen] = -np.inf
    top = scores.argmax(axis=1)  # best unseen item = the held-out one if the model learned
    hit = np.mean([top[u] in groups[u] for u in range(40)])
    assert hit >= 0.95, f"{name} put the user's own group first for only {hit:.0%} of users"


def test_merge_rows_replaces_rerun_models_and_keeps_order() -> None:
    old = [{"model": "random", "v": 1}, {"model": "als", "v": 1}, {"model": "bpr", "v": 1}]
    merged = merge_rows(old, [{"model": "als", "v": 2}], order=["random", "als", "bpr"])
    assert [(r["model"], r["v"]) for r in merged] == [("random", 1), ("als", 2), ("bpr", 1)]


def test_expand_grid_and_mlflow_key() -> None:
    combos = expand_grid({"seed": 1}, {"a": [1, 2], "b": ["x"]})
    assert combos == [{"seed": 1, "a": 1, "b": "x"}, {"seed": 1, "a": 2, "b": "x"}]
    assert expand_grid({"seed": 1}, {}) == [{"seed": 1}]
    assert mlflow_key("ndcg@10") == "ndcg_at_10"


def test_two_tower_grid_points_union_of_studies() -> None:
    from recsys.pipelines.two_tower import grid_points, run_label

    assert grid_points({"a": [1, 2]}) == [{"a": 1}, {"a": 2}]
    studies = [{}, {"a": [1, 2]}, {"b": [3]}, {"a": [1]}]  # duplicate {"a": 1} dropped
    assert grid_points(studies) == [{}, {"a": 1}, {"a": 2}, {"b": 3}]
    assert run_label({}) == "base" and run_label({"a": 1, "b": 2}) == "a=1, b=2"
