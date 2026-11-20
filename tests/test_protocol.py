from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.data.schema import IS_POSITIVE, ITEM_IDX, TIMESTAMP, USER_IDX
from recsys.eval import metrics
from recsys.eval.protocol import (
    build_eval_data,
    per_user_metrics,
    recommend_with_scores,
    summarize,
    topk_excluding_seen,
)


def _frame(rows: list[tuple[int, int, bool]]) -> pd.DataFrame:
    """rows: (user_idx, item_idx, is_positive)."""
    df = pd.DataFrame(rows, columns=[USER_IDX, ITEM_IDX, IS_POSITIVE])
    df[TIMESTAMP] = np.arange(len(df))
    return df


def test_topk_excludes_seen_before_cutting() -> None:
    scores = np.array([[0.9, 0.8, 0.1, 0.5], [0.1, 0.2, 0.3, 0.4]])
    top = topk_excluding_seen(scores, seen=[[0], [3, 2]], k=2)
    # Row 0: item 0 is seen, so the best unseen are 1 (0.8) and 3 (0.5).
    # Row 1: items 3 and 2 are seen -> 1, 0. We still return k items, not k - |seen|.
    assert top.tolist() == [[1, 3], [1, 0]]


def test_topk_pads_when_too_few_unseen_items() -> None:
    top = topk_excluding_seen(np.array([[0.3, 0.2, 0.1]]), seen=[[0, 1]], k=3)
    assert top.tolist() == [[2, metrics.PAD, metrics.PAD]]


def test_topk_does_not_mutate_input() -> None:
    scores = np.array([[0.9, 0.8]])
    topk_excluding_seen(scores, seen=[[0]], k=1)
    assert scores[0, 0] == 0.9


def test_build_eval_data() -> None:
    history = _frame([(0, 0, True), (0, 1, False), (1, 2, True)])
    target = _frame([(0, 3, True), (0, 4, False), (2, 5, True), (1, 6, False)])
    data = build_eval_data(history, target, n_items=7)

    assert data.user_idx.tolist() == [0, 2]  # user 1 has no positive target -> not evaluated
    assert [r.tolist() for r in data.relevant] == [[3], [5]]  # negatives aren't ground truth
    assert [s.tolist() for s in data.seen] == [[0, 1], []]  # seen includes disliked items
    assert data.is_warm.tolist() == [True, False]  # user 2 has no history: cold
    assert data.item_popularity.tolist() == [1, 1, 1, 0, 0, 0, 0]


def test_end_to_end_oracle_and_seen_guard() -> None:
    history = _frame([(0, 0, True), (1, 1, True)])
    target = _frame([(0, 2, True), (1, 3, True), (1, 4, True)])
    data = build_eval_data(history, target, n_items=5)

    # Oracle scores: the user's own target items score highest -> perfect metrics.
    def oracle(users: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(users), 5))
        for row, rel in enumerate(data.relevant[: len(users)]):
            scores[row, rel] = 1.0
        return scores

    recs = recommend_with_scores(oracle, data, k=2)
    result = summarize(recs, data, ks=(2,))
    assert result["all"]["recall@2"] == pytest.approx(1.0)
    assert result["all"]["ndcg@2"] == pytest.approx(1.0)
    assert result["cold"]["n_users"] == 0

    # A model that leaks a seen item into its list must be caught, not silently scored.
    bad = recs.copy()
    bad[0, 1] = 0
    with pytest.raises(ValueError, match="already-seen"):
        per_user_metrics(bad, data, ks=(2,))
