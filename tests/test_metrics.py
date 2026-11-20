"""Metrics checked against values computed by hand.

Worked example used throughout (user A):
    recs     = [3, 1, 7, 5, 9]
    relevant = {1, 5, 8}
    hits     = [0, 1, 0, 1, 0]   (hits at ranks 2 and 4)

    Recall@5   = 2 / 3
    HitRate@5  = 1
    MRR@5      = 1 / 2           (first hit at rank 2)
    DCG@5      = 1/log2(3) + 1/log2(5) = 0.6309298 + 0.4306766 = 1.0616064
    IDCG@5     = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.6309298 + 0.5 = 2.1309298
                 (min(|relevant|=3, k=5) = 3 ideal slots)
    NDCG@5     = 1.0616064 / 2.1309298 = 0.4981893

    At k=2: Recall 1/3, MRR 1/2, NDCG = 0.6309298 / (1 + 0.6309298) = 0.3868528
    At k=1: everything is 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from recsys.eval import metrics

A_RECS = [3, 1, 7, 5, 9]
A_REL = np.array([1, 5, 8])


def _single(recs: list[int], rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hits = metrics.hits_matrix(np.array([recs]), [rel])
    return hits, np.array([len(rel)])


def test_hits_matrix() -> None:
    hits, _ = _single(A_RECS, A_REL)
    assert hits.tolist() == [[False, True, False, True, False]]


@pytest.mark.parametrize(
    ("k", "recall", "hit_rate", "mrr", "ndcg"),
    [
        (5, 2 / 3, 1.0, 0.5, 0.4981893),
        (2, 1 / 3, 1.0, 0.5, 0.3868528),
        (1, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_worked_example(k: int, recall: float, hit_rate: float, mrr: float, ndcg: float) -> None:
    hits, n_rel = _single(A_RECS, A_REL)
    assert metrics.recall_at_k(hits, n_rel, k)[0] == pytest.approx(recall)
    assert metrics.hit_rate_at_k(hits, k)[0] == pytest.approx(hit_rate)
    assert metrics.mrr_at_k(hits, k)[0] == pytest.approx(mrr)
    assert metrics.ndcg_at_k(hits, n_rel, k)[0] == pytest.approx(ndcg, abs=1e-6)


def test_perfect_ranking_scores_one_even_with_more_relevant_than_k() -> None:
    # 4 relevant items, k=2, both slots relevant: NDCG must be exactly 1 because the
    # ideal list can also only place 2 items. (Classic bug: IDCG over all 4 -> 0.61.)
    hits, n_rel = _single([1, 2], np.array([1, 2, 3, 4]))
    assert metrics.ndcg_at_k(hits, n_rel, 2)[0] == pytest.approx(1.0)
    assert metrics.recall_at_k(hits, n_rel, 2)[0] == pytest.approx(0.5)  # recall caps at k/|rel|


def test_ndcg_rewards_higher_ranks() -> None:
    hit_first, n_rel = _single([1, 9, 9], np.array([1]))
    hit_last, _ = _single([9, 8, 1], np.array([1]))
    assert metrics.ndcg_at_k(hit_first, n_rel, 3)[0] == pytest.approx(1.0)
    assert metrics.ndcg_at_k(hit_last, n_rel, 3)[0] == pytest.approx(0.5)  # 1/log2(4)


def test_padding_never_counts() -> None:
    hits, n_rel = _single([metrics.PAD, metrics.PAD], np.array([metrics.PAD, 3]))
    assert not hits.any()
    assert metrics.mrr_at_k(hits, 2)[0] == 0.0


def test_multiple_users_vectorized() -> None:
    recs = np.array([A_RECS, [8, 0, 0, 0, 0]])
    rel = [A_REL, np.array([8])]
    hits = metrics.hits_matrix(recs, rel)
    n_rel = np.array([3, 1])
    np.testing.assert_allclose(metrics.recall_at_k(hits, n_rel, 5), [2 / 3, 1.0])
    np.testing.assert_allclose(metrics.mrr_at_k(hits, 5), [0.5, 1.0])
    np.testing.assert_allclose(metrics.ndcg_at_k(hits, n_rel, 5), [0.4981893, 1.0], atol=1e-6)


def test_k_out_of_range_raises() -> None:
    hits, n_rel = _single(A_RECS, A_REL)
    with pytest.raises(ValueError):
        metrics.recall_at_k(hits, n_rel, 6)


def test_catalog_coverage() -> None:
    recs = np.array([[1, 2, metrics.PAD], [2, 3, 4]])
    # k=2 -> {1, 2, 3} of 10 items; padding is ignored
    assert metrics.catalog_coverage(recs, n_items=10, k=2) == pytest.approx(0.3)
    assert metrics.catalog_coverage(recs, n_items=10, k=3) == pytest.approx(0.4)


def test_average_recommendation_popularity() -> None:
    popularity = np.array([0, 10, 20, 30])
    recs = np.array([[1, 2], [2, 3], [3, metrics.PAD]])
    # user means: (10+20)/2=15, (20+30)/2=25, 30 (pad ignored)
    np.testing.assert_allclose(
        metrics.average_recommendation_popularity(recs, popularity, 2), [15, 25, 30]
    )


def test_head_items_and_long_tail_share() -> None:
    popularity = np.array([5, 100, 1, 50, 0, 7, 3, 2, 9, 4])  # 10 items -> head = top 2
    head = metrics.head_items_by_popularity(popularity, head_frac=0.2)
    assert head.tolist() == [1, 3]
    recs = np.array([[1, 3, 0, 2], [1, 3, metrics.PAD, metrics.PAD]])
    np.testing.assert_allclose(metrics.long_tail_share(recs, head, 4), [0.5, 0.0])
