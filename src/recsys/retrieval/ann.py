"""FAISS indexes over L2-normalized item embeddings, plus recall/latency benchmarking.

Index types (all use inner product; on unit vectors that is cosine similarity):
  flat  exact brute force, O(n_items * d) per query. The reference everything is measured
        against, and genuinely the right choice for small catalogs.
  ivf   k-means partitions the items into `nlist` cells; a query scans only the `nprobe`
        nearest cells. Speed/recall knob: nprobe.
  hnsw  a navigable small-world graph; a query greedily walks the graph keeping `efSearch`
        candidates. Speed/recall knob: efSearch. Usually the best recall at low latency.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt

PAD = -1


def build_index(kind: str, vectors: npt.ArrayLike, **params: Any) -> faiss.Index:
    """Build a FAISS inner-product index.

    params: nlist / nprobe (ivf); M / ef_construction / ef_search (hnsw).
    """
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    d = x.shape[1]
    metric = faiss.METRIC_INNER_PRODUCT
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "ivf":
        # index_factory keeps the coarse quantizer alive inside the index (constructing
        # IndexIVFFlat by hand from Python risks the quantizer being garbage-collected).
        index = faiss.index_factory(d, f"IVF{params.get('nlist', 64)},Flat", metric)
        index.train(x)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, params.get("M", 32), metric)
        index.hnsw.efConstruction = params.get("ef_construction", 200)
    else:
        raise ValueError(f"Unknown index kind {kind!r}")
    index.add(x)
    set_search_params(index, **params)
    return index


def set_search_params(index: faiss.Index, **params: Any) -> None:
    if "nprobe" in params:
        faiss.extract_index_ivf(index).nprobe = params["nprobe"]
    if "ef_search" in params:
        index.hnsw.efSearch = params["ef_search"]


def search_excluding_seen(
    index: faiss.Index,
    queries: npt.ArrayLike,
    seen: Sequence[npt.ArrayLike],
    k: int,
    batch_size: int = 256,
) -> npt.NDArray[np.int64]:
    """Top-k item ids per query, skipping each user's already-seen items.

    ANN indexes can't apply a per-user exclusion list during the search, so we over-fetch
    k + |seen| candidates and filter afterwards (the same trick serving uses). Users are
    batched by similar |seen| so one heavy user doesn't force a huge fetch for everyone.
    """
    q = np.ascontiguousarray(queries, dtype=np.float32)
    n_items = index.ntotal
    out = np.full((len(q), k), PAD, dtype=np.int64)
    seen_sizes = np.array([len(s) for s in seen])
    order = np.argsort(seen_sizes, kind="stable")
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        fetch = int(min(n_items, k + seen_sizes[rows].max()))
        _, ids = index.search(q[rows], fetch)
        for row, cand in zip(rows, ids, strict=True):
            cand = cand[(cand != -1) & ~np.isin(cand, np.asarray(seen[row]))][:k]
            out[row, : len(cand)] = cand
    return out


def overlap_at_k(approx: npt.NDArray[np.int64], exact: npt.NDArray[np.int64]) -> float:
    """Mean |approx ∩ exact| / |exact| per query: the ANN's recall *relative to exact search*."""
    scores = []
    for a, e in zip(approx, exact, strict=True):
        e = e[e != PAD]
        if len(e):
            scores.append(np.isin(e, a).mean())
    return float(np.mean(scores))


@dataclass(frozen=True)
class LatencyStats:
    p50_us: float
    p95_us: float
    p99_us: float
    mean_us: float

    @classmethod
    def from_seconds(cls, seconds: npt.ArrayLike) -> LatencyStats:
        us = np.asarray(seconds) * 1e6
        p50, p95, p99 = np.percentile(us, [50, 95, 99])
        return cls(float(p50), float(p95), float(p99), float(us.mean()))


def single_query_latency(
    index: faiss.Index, queries: npt.ArrayLike, k: int, n_threads: int = 1, warmup: int = 20
) -> LatencyStats:
    """Latency of one query at a time (the online serving pattern), single-threaded."""
    q = np.ascontiguousarray(queries, dtype=np.float32)
    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(n_threads)
    try:
        for row in q[:warmup]:
            index.search(row[None, :], k)
        timings = []
        for row in q:
            start = time.perf_counter()
            index.search(row[None, :], k)
            timings.append(time.perf_counter() - start)
    finally:
        faiss.omp_set_num_threads(previous)
    return LatencyStats.from_seconds(timings)
