"""Maximal Marginal Relevance re-ranking for diversity.

Greedy: repeatedly pick the candidate maximizing

    lambda * relevance(i)  -  (1 - lambda) * max_{j already picked} sim(i, j)

lambda = 1 reproduces the ranker's order; smaller lambda trades relevance for items unlike
those already shown. Relevance is min-max normalized per user so lambda means the same
thing for every user regardless of the ranker's score scale.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from recsys.eval.metrics import PAD


def mmr_rerank(
    candidates: npt.NDArray[np.int64],
    relevance: npt.NDArray[np.floating],
    item_vectors: npt.NDArray[np.floating],
    k: int,
    lam: float,
) -> npt.NDArray[np.int64]:
    """candidates/relevance: [n_users, pool] (PAD-padded, relevance ignored there)."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    n_users, _ = candidates.shape
    out = np.full((n_users, k), PAD, dtype=np.int64)
    for u in range(n_users):
        valid = candidates[u] != PAD
        items = candidates[u][valid]
        if items.size == 0:
            continue
        rel = np.asarray(relevance[u][valid], dtype=np.float64)
        span = rel.max() - rel.min()
        rel = (rel - rel.min()) / span if span > 0 else np.ones_like(rel)
        vecs = item_vectors[items]
        sims = vecs @ vecs.T
        max_sim = np.full(items.size, -np.inf)  # nothing picked yet: no redundancy penalty
        available = np.ones(items.size, dtype=bool)
        for slot in range(min(k, items.size)):
            penalty = np.where(np.isfinite(max_sim), max_sim, 0.0)
            score = np.where(available, lam * rel - (1.0 - lam) * penalty, -np.inf)
            pick = int(np.argmax(score))
            out[u, slot] = items[pick]
            available[pick] = False
            max_sim = np.maximum(max_sim, sims[pick])
    return out
