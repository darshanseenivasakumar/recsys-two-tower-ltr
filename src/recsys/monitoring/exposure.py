"""What the system actually shows: catalog coverage and popularity concentration of served lists.

Offline coverage is a property of a model on a fixed user set; online it depends on who
shows up. A ranker that slowly collapses onto blockbusters shows up here (coverage down,
Gini and average popularity up) before it shows up in any accuracy metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from recsys.data.eda import gini


def exposure_summary(
    item_lists: Sequence[Sequence[int]],
    item_popularity: npt.ArrayLike,
    head_frac: float = 0.2,
) -> dict[str, Any]:
    """item_popularity: positives per item at the snapshot (defines head vs tail)."""
    pop = np.asarray(item_popularity, dtype=np.float64)
    n_items = len(pop)
    flat = np.fromiter((i for items in item_lists for i in items), dtype=np.int64)
    if flat.size == 0:
        return {"n_lists": len(item_lists), "n_impressions": 0}
    counts = np.bincount(flat, minlength=n_items)
    n_head = max(1, int(round(head_frac * n_items)))
    head = np.argsort(-pop, kind="stable")[:n_head]
    return {
        "n_lists": len(item_lists),
        "n_impressions": int(flat.size),
        "catalog_coverage": float((counts > 0).mean()),
        "exposure_gini": gini(counts),
        "avg_popularity": float(pop[flat].mean()),
        "long_tail_share": float(1.0 - np.isin(flat, head).mean()),
    }
