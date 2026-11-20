"""The interface every recommender implements, plus a cold-start fallback wrapper.

A model learns from a history DataFrame (canonical columns) and then scores all items for
a batch of users. Keeping the interface this small means one evaluation loop works for
every model, and the serving code in Phase 5 can call any of them the same way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.data.schema import IS_POSITIVE, USER_IDX

Scores = npt.NDArray[np.float32]
Users = npt.NDArray[np.int64]


class Recommender(ABC):
    """Base class. Subclasses are dataclasses whose fields are their hyperparameters."""

    name: str = "base"

    @abstractmethod
    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> Self:
        """Learn from `history` (all interactions; models decide what counts as signal)."""

    @abstractmethod
    def score(self, users: Users) -> Scores:
        """Return a dense [len(users), n_items] score matrix; higher = more recommended."""

    def params(self) -> dict[str, Any]:
        """Hyperparameters = the dataclass fields set in the constructor (not fitted state)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}  # type: ignore[arg-type]


@dataclass
class ColdStartFallback(Recommender):
    """Use `model` for users with >= 1 positive in history, `fallback` for everyone else.

    Why: a factor model has a user vector for every row, but for a user with no positive
    history that vector is untrained noise (or all zeros for KNN, which makes every
    score tie). Routing those users to popularity is what we'd do in production anyway.
    """

    model: Recommender
    fallback: Recommender
    is_warm_: npt.NDArray[np.bool_] = field(init=False, repr=False)

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.model.name

    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> Self:
        self.model.fit(history, n_users, n_items)
        self.fallback.fit(history, n_users, n_items)
        self.is_warm_ = np.zeros(n_users, dtype=bool)
        self.is_warm_[history.loc[history[IS_POSITIVE], USER_IDX].unique()] = True
        return self

    def score(self, users: Users) -> Scores:
        warm = self.is_warm_[users]
        if warm.all():
            return self.model.score(users)
        out = self.fallback.score(users).astype(np.float32, copy=True)
        if warm.any():
            out[warm] = self.model.score(users[warm])
        return out

    def params(self) -> dict[str, Any]:
        return self.model.params()
