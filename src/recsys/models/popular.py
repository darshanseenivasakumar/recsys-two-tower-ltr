"""Non-personalized baselines: random (the floor) and most-popular (the bar to clear)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.data.schema import IS_POSITIVE, ITEM_IDX, TIMESTAMP
from recsys.models.base import Recommender, Scores, Users

_SECONDS_PER_DAY = 86_400


@dataclass
class RandomRecommender(Recommender):
    """Uniform random scores. Any model scoring near this has learned nothing."""

    seed: int = 0
    n_items_: int = field(init=False, default=0, repr=False)
    rng_: np.random.Generator = field(init=False, repr=False)

    name = "random"

    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> Self:
        self.n_items_ = n_items
        self.rng_ = np.random.default_rng(self.seed)
        return self

    def score(self, users: Users) -> Scores:
        return self.rng_.random((len(users), self.n_items_), dtype=np.float32)


@dataclass
class MostPopular(Recommender):
    """Score every item by its number of positive interactions; same list for everyone.

    window_days: if set, count only positives from the last `window_days` days of the
    history. Popularity drifts, so "popular lately" can beat "popular ever".
    """

    window_days: float | None = None
    counts_: npt.NDArray[np.float32] = field(init=False, repr=False)

    name = "most_popular"

    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> Self:
        positives = history[history[IS_POSITIVE]]
        if self.window_days is not None:
            start = positives[TIMESTAMP].max() - self.window_days * _SECONDS_PER_DAY
            positives = positives[positives[TIMESTAMP] >= start]
        counts = np.bincount(positives[ITEM_IDX], minlength=n_items).astype(np.float32)
        # Break ties between equally popular items by all-time popularity (tiny epsilon),
        # so a short window doesn't leave the ranking to arbitrary index order.
        all_time = np.bincount(history.loc[history[IS_POSITIVE], ITEM_IDX], minlength=n_items)
        self.counts_ = counts + 1e-6 * all_time / max(1, all_time.max())
        return self

    def score(self, users: Users) -> Scores:
        return np.broadcast_to(self.counts_, (len(users), self.counts_.size))
