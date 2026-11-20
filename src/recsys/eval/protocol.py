"""The evaluation protocol: who is evaluated, against what, with what excluded.

Protocol (fixed for every model in this project):
  * Validation stage: fit on train, history = train,       targets = positives in val.
  * Test stage:       fit on train+val, history = train+val, targets = positives in test.
  * Evaluated users:  every user with >= 1 positive in the target period. Users with no
    history ("cold") are kept and reported separately; models must fall back for them.
  * Model selection: tune on the WARM segment of val. On ML-1M val is 55% cold users while
    test is 97.6% warm (no users join after Dec 2000), so all-user val metrics would
    tune for a different population than the one we are finally judged on.
  * Seen-item filtering: every item in the user's history (any rating, since they have
    watched it) is removed BEFORE taking the top-K. Filtering after the cut would
    return fewer than K items and penalize models that rank seen items highly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.data.schema import IS_POSITIVE, ITEM_IDX, USER_IDX
from recsys.data.split import SplitResult
from recsys.eval import metrics

Stage = Literal["val", "test"]
ScoreFn = Callable[[npt.NDArray[np.int64]], npt.NDArray[np.floating[Any]]]
"""Maps a batch of user indices [b] to a dense score matrix [b, n_items]."""

DEFAULT_KS: tuple[int, ...] = (10, 20, 50, 100, 200)


@dataclass(frozen=True)
class EvalData:
    """Everything needed to score one evaluation stage, aligned by position."""

    user_idx: npt.NDArray[np.int64]  # [n] users being evaluated, ascending
    relevant: list[npt.NDArray[np.int64]]  # ground-truth item_idx per user
    seen: list[npt.NDArray[np.int64]]  # history item_idx per user, to exclude
    is_warm: npt.NDArray[np.bool_]  # user has >= 1 history interaction
    item_popularity: npt.NDArray[np.int64]  # [n_items] interaction counts in history
    history: pd.DataFrame  # the interactions models may learn from

    @property
    def n_users(self) -> int:
        return len(self.user_idx)


def group_items(df: pd.DataFrame, users: npt.NDArray[np.int64]) -> list[npt.NDArray[np.int64]]:
    """For each user in `users`, the array of item_idx they have in `df` (empty if none)."""
    grouped = df.groupby(USER_IDX)[ITEM_IDX].agg(lambda s: s.to_numpy(dtype=np.int64))
    empty = np.empty(0, dtype=np.int64)
    return [grouped.get(u, empty) for u in users]


def build_eval_data(history: pd.DataFrame, target: pd.DataFrame, n_items: int) -> EvalData:
    """Assemble ground truth and exclusions for users with a positive in `target`."""
    positives = target[target[IS_POSITIVE]]
    users = np.sort(positives[USER_IDX].unique()).astype(np.int64)
    history_users = set(history[USER_IDX].unique())
    return EvalData(
        user_idx=users,
        relevant=group_items(positives, users),
        seen=group_items(history, users),
        is_warm=np.array([u in history_users for u in users], dtype=bool),
        item_popularity=np.bincount(history[ITEM_IDX], minlength=n_items).astype(np.int64),
        history=history,
    )


def eval_data_for_stage(split: SplitResult, stage: Stage, n_items: int) -> EvalData:
    if stage == "val":
        return build_eval_data(split.train, split.val, n_items)
    history = pd.concat([split.train, split.val], ignore_index=True)
    return build_eval_data(history, split.test, n_items)


def topk_excluding_seen(
    scores: npt.NDArray[np.floating[Any]], seen: Sequence[npt.ArrayLike], k: int
) -> npt.NDArray[np.int64]:
    """Top-k item indices per row after masking seen items to -inf.

    Uses argpartition (O(n_items)) then sorts only the k winners, instead of a full
    O(n_items log n_items) argsort per user. If a user has fewer than k unseen items the
    row is padded with PAD.
    """
    scores = np.array(scores, dtype=np.float64, copy=True)
    n_rows, n_items = scores.shape
    for row, items in enumerate(seen):
        scores[row, np.asarray(items, dtype=np.int64)] = -np.inf

    k_eff = min(k, n_items)
    part = np.argpartition(-scores, k_eff - 1, axis=1)[:, :k_eff]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    top = np.take_along_axis(part, order, axis=1)
    top_scores = np.take_along_axis(part_scores, order, axis=1)

    out = np.full((n_rows, k), metrics.PAD, dtype=np.int64)
    out[:, :k_eff] = np.where(np.isneginf(top_scores), metrics.PAD, top)
    return out


def recommend_with_scores(
    score_fn: ScoreFn, data: EvalData, k: int, batch_size: int = 1024
) -> npt.NDArray[np.int64]:
    """Run a dense-scoring model over all evaluated users in batches; return top-k recs."""
    chunks = []
    for start in range(0, data.n_users, batch_size):
        users = data.user_idx[start : start + batch_size]
        scores = score_fn(users)
        chunks.append(topk_excluding_seen(scores, data.seen[start : start + batch_size], k))
    return np.vstack(chunks)


def per_user_metrics(
    recs: npt.NDArray[np.int64], data: EvalData, ks: Sequence[int] = DEFAULT_KS
) -> pd.DataFrame:
    """One row per evaluated user, one column per metric@k."""
    if recs.shape[0] != data.n_users:
        raise ValueError("recs rows must align with data.user_idx")
    for row, items in enumerate(data.seen):
        if np.isin(recs[row], items).any():
            raise ValueError(f"User {data.user_idx[row]} was recommended an already-seen item")

    hits = metrics.hits_matrix(recs, data.relevant)
    n_rel = np.array([len(r) for r in data.relevant], dtype=np.int64)
    head = metrics.head_items_by_popularity(data.item_popularity)
    cols: dict[str, Any] = {"user_idx": data.user_idx, "is_warm": data.is_warm}
    for k in ks:
        if k > recs.shape[1]:
            continue
        cols[f"recall@{k}"] = metrics.recall_at_k(hits, n_rel, k)
        cols[f"ndcg@{k}"] = metrics.ndcg_at_k(hits, n_rel, k)
        cols[f"hit_rate@{k}"] = metrics.hit_rate_at_k(hits, k)
        cols[f"mrr@{k}"] = metrics.mrr_at_k(hits, k)
        cols[f"arp@{k}"] = metrics.average_recommendation_popularity(recs, data.item_popularity, k)
        cols[f"long_tail@{k}"] = metrics.long_tail_share(recs, head, k)
    return pd.DataFrame(cols)


def summarize(
    recs: npt.NDArray[np.int64], data: EvalData, ks: Sequence[int] = DEFAULT_KS
) -> dict[str, dict[str, float]]:
    """Mean metrics for all / warm / cold users, plus catalog coverage."""
    table = per_user_metrics(recs, data, ks)
    metric_cols = [c for c in table.columns if "@" in c]
    out: dict[str, dict[str, float]] = {}
    segments = {"all": table, "warm": table[table.is_warm], "cold": table[~table.is_warm]}
    for name, seg in segments.items():
        stats = {c: float(seg[c].mean()) for c in metric_cols} if len(seg) else {}
        stats["n_users"] = float(len(seg))
        out[name] = stats
    n_items = len(data.item_popularity)
    for k in ks:
        if k <= recs.shape[1]:
            out["all"][f"coverage@{k}"] = metrics.catalog_coverage(recs, n_items, k)
    return out
