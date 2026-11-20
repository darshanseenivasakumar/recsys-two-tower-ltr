"""Reusable statistics for exploratory analysis (the notebook only calls these and plots).

Kept out of the notebook so they are tested and can be re-run on ML-25M unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.data.schema import IS_POSITIVE, ITEM_ID, RATING, TIMESTAMP, USER_ID


def gini(counts: npt.ArrayLike) -> float:
    """Gini coefficient of a count distribution: 0 = perfectly even, ->1 = one item has all.

    With x sorted ascending and 1-based rank i:  G = 2*sum(i*x_i) / (n*sum(x)) - (n+1)/n
    """
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n, total = x.size, x.sum()
    if n == 0 or total == 0:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float(2.0 * (ranks * x).sum() / (n * total) - (n + 1) / n)


def head_share(
    counts: npt.ArrayLike, fracs: Sequence[float] = (0.01, 0.1, 0.2)
) -> dict[float, float]:
    """Share of all interactions captured by the most popular `frac` of items."""
    x = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
    total = x.sum()
    return {f: float(x[: max(1, int(round(f * x.size)))].sum() / total) for f in fracs}


def overview(interactions: pd.DataFrame, n_items_catalog: int) -> dict[str, Any]:
    n_users = interactions[USER_ID].nunique()
    n_items = interactions[ITEM_ID].nunique()
    return {
        "interactions": len(interactions),
        "users": n_users,
        "items_with_interactions": n_items,
        "items_in_catalog": n_items_catalog,
        "density": len(interactions) / (n_users * n_items),
        "positive_rate": float(interactions[IS_POSITIVE].mean()),
        "mean_rating": float(interactions[RATING].mean()),
        "first_interaction": pd.to_datetime(interactions[TIMESTAMP].min(), unit="s"),
        "last_interaction": pd.to_datetime(interactions[TIMESTAMP].max(), unit="s"),
    }


def count_distribution(counts: pd.Series) -> dict[str, float]:
    q = counts.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "min": float(counts.min()),
        **{f"p{int(k * 100)}": float(v) for k, v in q.items()},
        "max": float(counts.max()),
        "mean": float(counts.mean()),
    }


def user_activity_span_days(interactions: pd.DataFrame) -> pd.Series:
    """Days between each user's first and last interaction."""
    ts = interactions.groupby(USER_ID)[TIMESTAMP]
    return (ts.max() - ts.min()) / 86_400
