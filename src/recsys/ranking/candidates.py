"""Candidate generation: the union of several retrievers' top-k lists.

Each source contributes its rank and score as features, so the ranker learns how much to
trust each retriever, and whether agreement between them (n_sources = 2) matters. An item
a source did not retrieve gets rank = k + 1 and score = NaN (LightGBM routes NaN itself).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.eval.metrics import PAD
from recsys.eval.protocol import EvalData, topk_excluding_seen

ScoreFn = Callable[[npt.NDArray[np.int64]], npt.NDArray[np.floating]]
IntArray = npt.NDArray[np.int64]


def topk_with_scores(
    score_fn: ScoreFn, data: EvalData, k: int, batch_size: int = 1024
) -> tuple[IntArray, npt.NDArray[np.float64]]:
    """Top-k unseen items per evaluated user and their scores (NaN where padded)."""
    recs, scores = [], []
    for start in range(0, data.n_users, batch_size):
        users = data.user_idx[start : start + batch_size]
        dense = score_fn(users)
        top = topk_excluding_seen(dense, data.seen[start : start + batch_size], k)
        vals = np.take_along_axis(dense, np.maximum(top, 0), axis=1).astype(np.float64)
        recs.append(top)
        scores.append(np.where(top == PAD, np.nan, vals))
    return np.vstack(recs), np.vstack(scores)


@dataclass(frozen=True)
class Candidates:
    """Pairs sorted by (row, item). row indexes EvalData.user_idx, not user_idx itself."""

    rows: IntArray
    items: IntArray
    retrieval: dict[str, npt.NDArray[np.float64]]  # per pair: <src>_score, <src>_rank, ...

    def __len__(self) -> int:
        return len(self.rows)


def merge_sources(sources: Mapping[str, tuple[IntArray, npt.NDArray[np.float64]]]) -> Candidates:
    """Outer-join each source's (recs, scores) [n_rows, k] into one candidate list per row.

    Pure numpy (sorted (row, item) keys + searchsorted): it runs on every API request,
    where a pandas outer merge of two ~200-row frames cost ~10 ms, almost all overhead.
    """
    if not sources:
        raise ValueError("need at least one candidate source")
    flat = {}
    for name, (recs, scores) in sources.items():
        n_rows, k = recs.shape
        items = recs.ravel().astype(np.int64)
        keep = items != PAD
        flat[name] = (
            np.repeat(np.arange(n_rows, dtype=np.int64), k)[keep],
            items[keep],
            np.tile(np.arange(1, k + 1, dtype=np.float64), n_rows)[keep],
            np.asarray(scores, dtype=np.float64).ravel()[keep],
            k,
        )
    width = 1 + max((int(f[1].max()) for f in flat.values() if f[1].size), default=0)
    keys = np.unique(np.concatenate([f[0] * width + f[1] for f in flat.values()]))  # row, item
    retrieval: dict[str, npt.NDArray[np.float64]] = {}
    present = np.zeros(len(keys))
    for name, (rows, items, ranks, scores, k) in flat.items():
        pos = np.searchsorted(keys, rows * width + items)
        rank = np.full(len(keys), float(k + 1))  # not retrieved by this source
        score = np.full(len(keys), np.nan)
        rank[pos], score[pos] = ranks, scores
        present[pos] += 1
        retrieval[f"{name}_rank"] = rank
        retrieval[f"{name}_score"] = score
    retrieval["n_sources"] = present
    return Candidates(rows=keys // width, items=keys % width, retrieval=retrieval)


def best_rank(retrieval: Mapping[str, npt.ArrayLike]) -> npt.NDArray[np.float64]:
    """Each pair's best rank over the sources (k + 1 where a source didn't retrieve it)."""
    ranks = [np.asarray(v, dtype=np.float64) for k, v in retrieval.items() if k.endswith("_rank")]
    return np.minimum.reduce(ranks)


def within_cap(retrieval: Mapping[str, npt.ArrayLike], cap: int | None) -> npt.NDArray[np.bool_]:
    """Pairs the ranker scores: some source ranked them in its top `cap` (None = all).

    The ranker is LightGBM's cost driver and is linear in rows; candidates every retriever
    ranks low almost never reach the top 10, so they are skipped before featurization.
    Kept pairs keep every feature value (ranks and n_sources still come from the full top-k
    lists), so their scores are exactly what the uncapped ranker would give them.
    """
    rank = best_rank(retrieval)
    return np.ones(len(rank), dtype=bool) if cap is None else rank <= cap


def label_candidates(candidates: Candidates, data: EvalData) -> npt.NDArray[np.int64]:
    """1 if the candidate is one of the row's held-out positives, else 0."""
    rel = pd.DataFrame(
        {
            "row": np.repeat(np.arange(data.n_users), [len(r) for r in data.relevant]),
            "item": np.concatenate([*data.relevant, np.empty(0, np.int64)]),
            "label": 1,
        }
    )
    pairs = pd.DataFrame({"row": candidates.rows, "item": candidates.items})
    return pairs.merge(rel, on=["row", "item"], how="left")["label"].fillna(0).to_numpy(np.int64)


def score_pairs(
    score_fn: ScoreFn,
    data: EvalData,
    rows: IntArray,
    items: IntArray,
    batch_size: int = 1024,
) -> npt.NDArray[np.float64]:
    """A retriever's score for every (row, item) candidate, retrieved by it or not.

    Ranks are only defined for items a source returned, but its score exists for any item;
    giving the ranker both retrievers' scores on the whole union is strictly more signal.
    """
    out = np.empty(len(rows), dtype=np.float64)
    for start in range(0, data.n_users, batch_size):
        stop = min(start + batch_size, data.n_users)
        dense = score_fn(data.user_idx[start:stop])
        sel = (rows >= start) & (rows < stop)
        out[sel] = dense[rows[sel] - start, items[sel]]
    return out


def retrieval_features(
    candidates: Candidates,
    score_fns: Mapping[str, ScoreFn],
    data: EvalData,
    has_vector: Mapping[str, npt.NDArray[np.bool_]] | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Per-pair retriever features, shared by the training pipeline and the API.

    Each source's rank (k + 1 if it did not retrieve the item) and `n_sources` come from
    the merge; each source's score is computed for EVERY candidate. `has_vector[name]`
    is a per-row mask: False means the source has no learned representation for that
    row's user and scored with a fallback instead (BPR -> raw popularity counts, a
    different unit from a dot product). Those scores become NaN, so "BPR doesn't know
    this user" is explicit rather than a score 100x larger than any real one.
    """
    out = dict(candidates.retrieval)
    for name, fn in score_fns.items():
        scores = score_pairs(fn, data, candidates.rows, candidates.items)
        if has_vector is not None and name in has_vector:
            scores = np.where(np.asarray(has_vector[name])[candidates.rows], scores, np.nan)
        out[f"{name}_score"] = scores
    return out


def top_scored(
    rows: IntArray, items: IntArray, scores: npt.ArrayLike, n_rows: int, k: int
) -> tuple[IntArray, npt.NDArray[np.float64]]:
    """Each row's top-k candidates by descending score: (items, scores), PAD/NaN-padded."""
    s = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((items, -s, rows))  # by row, then score desc, then item (stable ties)
    r, it, sc = rows[order], items[order], s[order]
    starts = np.searchsorted(r, np.arange(n_rows))
    position = np.arange(len(r)) - starts[r]
    keep = position < k
    out = np.full((n_rows, k), PAD, dtype=np.int64)
    vals = np.full((n_rows, k), np.nan)
    out[r[keep], position[keep]] = it[keep]
    vals[r[keep], position[keep]] = sc[keep]
    return out, vals


def scores_to_recs(
    rows: IntArray, items: IntArray, scores: npt.ArrayLike, n_rows: int, k: int
) -> IntArray:
    """Order each row's candidates by descending score -> [n_rows, k] recs, PAD-padded."""
    return top_scored(rows, items, scores, n_rows, k)[0]
