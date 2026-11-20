"""Ranking queries: (user, decision time) pairs, the unit the ranker trains and is judged on.

Offline evaluation asks each user once, at the snapshot cutoff. Serving asks whenever the
user opens the app, usually mid-session, right after interactions the retrievers were
never fit on. A ranker trained only on cutoff queries sees `days_since_last` measured from
the end of the history window (mean ~15 days in training) but serves requests made minutes
after the last rating (mean 0.6 days in the replay): a point-in-time mismatch that the
parity test cannot catch, because it compares offline and online at the same moment.

So a window (history before its cutoff, label period after) yields two kinds of query:

* cutoff query   decision time = the history cutoff; history = the window history;
                 labels = every positive in the label period. This IS the offline protocol.
* session query  decision time t = the timestamp of a random label-period event of the
                 user; history = window history + the user's label-period events strictly
                 before t; labels = the user's positives at or after t. The retrievers and
                 item statistics stay frozen at the window cutoff, exactly like a served
                 bundle, while the user's history is fresh, exactly like Redis.

Session times are drawn only from events that still have a positive at or after them
(LambdaRank learns nothing from a query without one, so sampling the others is waste).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.data.schema import IS_POSITIVE, ITEM_IDX, RATING, TIMESTAMP, USER_IDX
from recsys.eval.protocol import EvalData

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Events:
    """One user's interactions, sorted by (timestamp, item)."""

    items: IntArray
    ratings: FloatArray
    timestamps: FloatArray
    positive: BoolArray

    def __len__(self) -> int:
        return len(self.items)

    @classmethod
    def empty(cls) -> Events:
        return cls(np.empty(0, np.int64), np.empty(0), np.empty(0), np.empty(0, dtype=bool))

    def before(self, t: float) -> Events:
        keep = self.timestamps < t
        return Events(
            self.items[keep], self.ratings[keep], self.timestamps[keep], self.positive[keep]
        )

    def concat(self, other: Events) -> Events:
        return Events(
            np.concatenate([self.items, other.items]),
            np.concatenate([self.ratings, other.ratings]),
            np.concatenate([self.timestamps, other.timestamps]),
            np.concatenate([self.positive, other.positive]),
        )

    def liked(self) -> IntArray:
        """Positive items oldest -> newest (the two-tower history encoder's input order)."""
        return self.items[self.positive]


def events_by_user(df: pd.DataFrame) -> dict[int, Events]:
    df = df.sort_values([USER_IDX, TIMESTAMP, ITEM_IDX], kind="stable")
    u = df[USER_IDX].to_numpy(np.int64)
    cols = (
        df[ITEM_IDX].to_numpy(np.int64),
        df[RATING].to_numpy(np.float64),
        df[TIMESTAMP].to_numpy(np.float64),
        df[IS_POSITIVE].to_numpy(bool),
    )
    starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]]) if len(u) else np.empty(0, int)
    ends = np.r_[starts[1:], len(u)]
    return {int(u[s]): Events(*(c[s:e] for c in cols)) for s, e in zip(starts, ends, strict=True)}


@dataclass(frozen=True)
class Queries:
    """Row q of every array (and of `data`) is query q; queries are grouped by user."""

    user: IntArray  # user_idx per query
    as_of: FloatArray  # decision time (unix seconds)
    at_cutoff: BoolArray  # True: the offline-protocol query
    history: list[Events]  # what the user had done before the decision
    data: EvalData  # user_idx = query ids 0..n-1; relevant = labels; seen = history items

    @property
    def n(self) -> int:
        return len(self.user)

    @property
    def personalized(self) -> BoolArray:
        """Queries the API would personalize (>= 1 positive before the decision)."""
        return np.array([h.positive.any() for h in self.history], dtype=bool)

    def meta(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "user": self.user,
                "as_of": self.as_of,
                "at_cutoff": self.at_cutoff,
                "personalized": self.personalized,
                "is_warm": self.data.is_warm,
                "n_relevant": [len(r) for r in self.data.relevant],
            }
        )


def build_queries(
    history: pd.DataFrame,
    target: pd.DataFrame,
    n_items: int,
    sessions_per_user: int,
    rng: np.random.Generator,
) -> Queries:
    """One cutoff query plus up to `sessions_per_user` session queries per target user.

    Target users are those with >= 1 positive in `target` (the evaluation protocol's
    rule), in ascending user order, so with `sessions_per_user=0` the queries are exactly
    `build_eval_data(history, target)`'s users, labels and seen sets.
    """
    cutoff = float(history[TIMESTAMP].max())
    if len(target) and float(target[TIMESTAMP].min()) <= cutoff:
        raise ValueError("label period must start after the history cutoff")
    past, future = events_by_user(history), events_by_user(target)
    users = np.sort(target.loc[target[IS_POSITIVE], USER_IDX].unique()).astype(np.int64)

    q_user: list[int] = []
    q_time: list[float] = []
    q_cut: list[bool] = []
    q_hist: list[Events] = []
    q_rel: list[IntArray] = []
    for u in users.tolist():
        h, f = past.get(u, Events.empty()), future[u]
        times = [cutoff]
        if sessions_per_user > 0:
            last_like = f.timestamps[f.positive].max()
            eligible = np.flatnonzero(f.timestamps <= last_like)
            size = min(sessions_per_user, len(eligible))
            picks = rng.choice(eligible, size=size, replace=False)
            times += np.unique(f.timestamps[picks]).tolist()  # same-second picks collapse
        for i, t in enumerate(times):
            labels = f.positive & (f.timestamps >= t)
            q_user.append(u)
            q_time.append(float(t))
            q_cut.append(i == 0)
            q_hist.append(h if i == 0 else h.concat(f.before(t)))
            q_rel.append(np.unique(f.items[labels]))

    n = len(q_user)
    data = EvalData(
        user_idx=np.arange(n, dtype=np.int64),
        relevant=q_rel,
        seen=[np.unique(h.items) for h in q_hist],
        is_warm=np.array([len(h) > 0 for h in q_hist], dtype=bool),
        item_popularity=np.bincount(history[ITEM_IDX], minlength=n_items).astype(np.int64),
        history=history,
    )
    return Queries(
        user=np.array(q_user, dtype=np.int64),
        as_of=np.array(q_time, dtype=np.float64),
        at_cutoff=np.array(q_cut, dtype=bool),
        history=q_hist,
        data=data,
    )
