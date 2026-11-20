"""Top-K recommendation metrics, implemented from scratch with numpy.

Conventions
-----------
recs:      int array [n_users, K_max]. Row u is user u's ranked recommendations, best first,
           as item indices. Rows shorter than K_max are padded with PAD (-1).
relevant:  list of n_users int arrays, the items each user actually interacted with
           positively in the evaluation period (the "ground truth").
hits:      bool array [n_users, K_max]; hits[u, r] is True if recs[u, r] is relevant to u.

Accuracy metrics are computed from `hits` alone, so they are cheap and easy to verify.
All "per-user" functions return one value per user; average them yourself (or use
`summarize`) so you can also slice by user segment (warm vs. cold, activity level, ...).

Binary relevance throughout: an item is either relevant (1) or not (0).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

PAD = -1

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


def hits_matrix(recs: npt.ArrayLike, relevant: Sequence[npt.ArrayLike]) -> BoolArray:
    """hits[u, r] = recs[u, r] is in relevant[u]. Padding never counts as a hit."""
    recs = np.asarray(recs)
    if recs.ndim != 2:
        raise ValueError(f"recs must be 2-D [n_users, K], got shape {recs.shape}")
    if len(relevant) != recs.shape[0]:
        raise ValueError(f"{recs.shape[0]} rows of recs but {len(relevant)} relevant sets")
    hits = np.zeros(recs.shape, dtype=bool)
    for u, rel in enumerate(relevant):
        hits[u] = np.isin(recs[u], np.asarray(rel))
    hits &= recs != PAD
    return hits


def _n_relevant(relevant: Sequence[npt.ArrayLike]) -> IntArray:
    n_rel = np.array([len(np.unique(np.asarray(r))) for r in relevant], dtype=np.int64)
    if (n_rel == 0).any():
        raise ValueError("Every evaluated user needs >= 1 relevant item; filter users first")
    return n_rel


def _check_k(hits: BoolArray, k: int) -> None:
    if not 1 <= k <= hits.shape[1]:
        raise ValueError(f"k={k} must be in [1, {hits.shape[1]}] (number of recommendations)")


def recall_at_k(hits: BoolArray, n_relevant: IntArray, k: int) -> FloatArray:
    """|top-k ∩ relevant| / |relevant|.

    Note: if a user has more than k relevant items, the maximum achievable recall is
    k / |relevant| < 1. That is standard Recall@K; compare models, not absolute values.
    """
    _check_k(hits, k)
    return hits[:, :k].sum(axis=1) / n_relevant


def hit_rate_at_k(hits: BoolArray, k: int) -> FloatArray:
    """1.0 if at least one relevant item appears in the top-k, else 0.0."""
    _check_k(hits, k)
    return hits[:, :k].any(axis=1).astype(np.float64)


def mrr_at_k(hits: BoolArray, k: int) -> FloatArray:
    """1 / (rank of the first relevant item), 0 if none in the top-k. Ranks are 1-based."""
    _check_k(hits, k)
    top = hits[:, :k]
    first = top.argmax(axis=1)  # index of the first True (0 if there is none)
    return np.where(top.any(axis=1), 1.0 / (first + 1), 0.0)


def ndcg_at_k(hits: BoolArray, n_relevant: IntArray, k: int) -> FloatArray:
    """Normalized Discounted Cumulative Gain with binary relevance.

    DCG@k  = sum over ranks r=1..k of rel_r / log2(r + 1)
    IDCG@k = DCG of a perfect list = sum_{r=1}^{min(|relevant|, k)} 1 / log2(r + 1)
    NDCG@k = DCG@k / IDCG@k, in [0, 1].

    Rewards putting relevant items near the top: a hit at rank 1 is worth 1.0, at rank 2
    0.63, at rank 10 0.29.
    """
    _check_k(hits, k)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # ranks 1..k -> log2(2..k+1)
    dcg = hits[:, :k].astype(np.float64) @ discounts
    ideal_cumulative = np.cumsum(discounts)  # ideal_cumulative[j] = IDCG with j+1 hits
    idcg = ideal_cumulative[np.minimum(n_relevant, k) - 1]
    return dcg / idcg


# ------------------------------------------------------------------ beyond-accuracy


def catalog_coverage(recs: npt.ArrayLike, n_items: int, k: int) -> float:
    """Fraction of the catalog that appears in at least one user's top-k.

    A model can score well on accuracy by recommending the same 50 blockbusters to
    everyone; coverage exposes that.
    """
    top = np.asarray(recs)[:, :k]
    return float(np.unique(top[top != PAD]).size / n_items)


def average_recommendation_popularity(
    recs: npt.ArrayLike, item_popularity: npt.ArrayLike, k: int
) -> FloatArray:
    """Per user: mean training popularity (interaction count) of the recommended items.

    Higher = more popularity-biased. Compare against the popularity of what users
    actually consumed to see whether a model amplifies popularity bias.
    """
    top = np.asarray(recs)[:, :k]
    pop = np.asarray(item_popularity, dtype=np.float64)
    valid = top != PAD
    values = np.where(valid, pop[np.where(valid, top, 0)], 0.0)
    counts = valid.sum(axis=1)
    return np.divide(values.sum(axis=1), counts, out=np.zeros(len(top)), where=counts > 0)


def long_tail_share(recs: npt.ArrayLike, head_items: npt.ArrayLike, k: int) -> FloatArray:
    """Per user: fraction of recommended slots filled by items outside the popular head."""
    top = np.asarray(recs)[:, :k]
    valid = top != PAD
    tail = valid & ~np.isin(top, np.asarray(head_items))
    counts = valid.sum(axis=1)
    return np.divide(tail.sum(axis=1), counts, out=np.zeros(len(top)), where=counts > 0)


def head_items_by_popularity(item_popularity: npt.ArrayLike, head_frac: float = 0.2) -> IntArray:
    """Indices of the most popular `head_frac` of items (ties broken by item index)."""
    pop = np.asarray(item_popularity)
    n_head = max(1, int(round(head_frac * pop.size)))
    order = np.lexsort((np.arange(pop.size), -pop))  # by popularity desc, then index asc
    return np.sort(order[:n_head])


def intra_list_diversity(recs: npt.ArrayLike, item_vectors: npt.ArrayLike, k: int) -> FloatArray:
    """Per user: mean pairwise (1 - cosine) among the top-k items. NaN if < 2 valid items.

    Higher = more varied list. `item_vectors` need not be normalized.
    """
    top = np.asarray(recs)[:, :k]
    vecs = np.asarray(item_vectors, dtype=np.float64)
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    out = np.full(len(top), np.nan)
    for u, row in enumerate(top):
        items = row[row != PAD]
        n = items.size
        if n < 2:
            continue
        sims = vecs[items] @ vecs[items].T
        mean_sim = (sims.sum() - np.trace(sims)) / (n * (n - 1))
        out[u] = 1.0 - mean_sim
    return out
