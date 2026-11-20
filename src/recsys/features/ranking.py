"""Features for the ranker: one row per (user, candidate item) pair.

Everything is computed from interactions strictly before a cutoff (the end of the
history window), so a training row never contains information from its own label period.

Three pieces, split by how often they change, which is also how serving stores them:

* `ItemTable`   per-item statistics as of the cutoff. Built once per batch snapshot and
                loaded into memory by the API (item stats move slowly).
* `UserState`   per-user summary built by `user_state()` from that user's raw interaction
                list. This is the ONE implementation: offline it runs over every user's
                history, online it runs over the user's list in Redis after each new
                interaction. One function -> no training-serving skew (tests enforce it).
* `feature_columns()`  pure function: (stacked user states, item table, candidates,
                retriever scores) -> {feature name: column}. Identical call offline and
                online; `compute_features()` wraps it in a DataFrame (training tables),
                `feature_matrix()` stacks it into the float matrix the API hands LightGBM
                (a 47-column DataFrame costs ~1 ms per request in pandas overhead alone).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.data.schema import GENRES, IS_POSITIVE, ITEM_IDX, RATING, TIMESTAMP, USER_IDX, YEAR
from recsys.features.encoders import UserFeatures

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

SECONDS_PER_DAY = 86_400.0
RECENT_N = 20  # liked items kept for the embedding-similarity features
RATE_SMOOTHING = 10.0  # pseudo-counts pulling rare items' rates toward the global rate
GENRE_SMOOTHING = 5.0  # pseudo-positives pulling a short history toward the global genre mix
POP_WINDOWS_DAYS = (7, 30)

CATEGORICAL_FEATURES = ["user_gender", "user_age", "user_occupation"]


@dataclass(frozen=True)
class ItemTable:
    """Per-item statistics as of `cutoff`, indexed by item_idx."""

    numeric: pd.DataFrame  # [n_items, item feature columns]
    genres: npt.NDArray[np.float64]  # [n_items, G] multi-hot
    genre_names: list[str]
    log_pop: FloatArray  # log1p(#positives in history); also used to build user states
    year: FloatArray  # release year, NaN if unknown
    global_genre_share: FloatArray  # [G] share of all positives carrying each genre
    cutoff: float  # unix seconds: the end of the history window

    @classmethod
    def from_history(cls, history: pd.DataFrame, items: pd.DataFrame) -> ItemTable:
        items = items.sort_values(ITEM_IDX)
        n_items = len(items)
        it = history[ITEM_IDX].to_numpy(dtype=np.int64)
        ts = history[TIMESTAMP].to_numpy(dtype=np.float64)
        rating = history[RATING].to_numpy(dtype=np.float64)
        pos = history[IS_POSITIVE].to_numpy(dtype=bool)
        cutoff = float(ts.max()) if len(ts) else 0.0

        n = np.bincount(it, minlength=n_items).astype(np.float64)
        n_pos = np.bincount(it, weights=pos, minlength=n_items)
        rating_sum = np.bincount(it, weights=rating, minlength=n_items)
        global_pos_rate = float(pos.mean()) if len(pos) else 0.0
        global_rating = float(rating.mean()) if len(rating) else 0.0

        year = items[YEAR].astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan)
        names = sorted({g for genres in items[GENRES] for g in genres})
        col = {g: j for j, g in enumerate(names)}
        genres = np.zeros((n_items, len(names)))
        for row, item_genres in enumerate(items[GENRES]):
            genres[row, [col[g] for g in item_genres]] = 1.0
        liked = it[pos]
        global_share = genres[liked].mean(0) if liked.size else np.zeros(len(names))

        log_pop = np.log1p(n_pos)
        # No "item age" (cutoff - first rating): MovieLens-1M was collected in one burst, so
        # almost every item's first rating is in mid-2000 and its age is really "days since
        # the dataset began", a calendar clock the drift monitor flagged on day 0 (PSI 7.2).
        # Release year carries the genuine item-age signal and does not move with time.
        numeric = {
            "item_log_n_ratings": np.log1p(n),
            "item_log_pop": log_pop,
            "item_pos_rate": (n_pos + RATE_SMOOTHING * global_pos_rate) / (n + RATE_SMOOTHING),
            "item_mean_rating": (rating_sum + RATE_SMOOTHING * global_rating)
            / (n + RATE_SMOOTHING),
            "item_year": year,
        }
        for days in POP_WINDOWS_DAYS:
            recent = pos & (ts >= cutoff - days * SECONDS_PER_DAY)
            numeric[f"item_log_pop_{days}d"] = np.log1p(
                np.bincount(it[recent], minlength=n_items).astype(np.float64)
            )
        return cls(
            numeric=pd.DataFrame(numeric),
            genres=genres,
            genre_names=names,
            log_pop=log_pop,
            year=year,
            global_genre_share=global_share,
            cutoff=cutoff,
        )


@dataclass(frozen=True)
class UserState:
    """One user's summary as of the table's cutoff. NaN = undefined (e.g. no history)."""

    n_ratings: float
    n_pos: float
    mean_rating: float
    days_since_last: float
    span_days: float
    mean_log_pop: float  # how mainstream the user's liked items are
    mean_year: float  # mean release year of liked items
    genre_share: FloatArray  # [G] smoothed share of the user's positives per genre
    recent: IntArray  # last RECENT_N liked item_idx, oldest -> newest


USER_SCALARS = [
    "n_ratings",
    "n_pos",
    "mean_rating",
    "days_since_last",
    "span_days",
    "mean_log_pop",
    "mean_year",
]


def user_state(
    items: npt.ArrayLike,
    ratings: npt.ArrayLike,
    timestamps: npt.ArrayLike,
    is_positive: npt.ArrayLike,
    table: ItemTable,
    recent_n: int = RECENT_N,
    as_of: float | None = None,
) -> UserState:
    """THE user-feature implementation, shared by offline training and online serving.

    Inputs are the user's raw interactions (any order). `as_of` is the decision time that
    time features are measured from: offline it is the window cutoff (the default); online
    it is the request time, so `days_since_last` keeps meaning "days before the decision"
    after the user interacts again (with a frozen cutoff it would go negative).
    """
    now = table.cutoff if as_of is None else float(as_of)
    it = np.asarray(items, dtype=np.int64)
    ts = np.asarray(timestamps, dtype=np.float64)
    order = np.lexsort((it, ts))  # time, then item: deterministic under timestamp ties
    it, ts = it[order], ts[order]
    rt = np.asarray(ratings, dtype=np.float64)[order]
    pos = np.asarray(is_positive, dtype=bool)[order]
    if it.size == 0:
        return UserState(
            0.0,
            0.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            table.global_genre_share.copy(),
            np.empty(0, dtype=np.int64),
        )
    liked = it[pos]
    counts = table.genres[liked].sum(0)
    share = (counts + GENRE_SMOOTHING * table.global_genre_share) / (liked.size + GENRE_SMOOTHING)
    years = table.year[liked]
    years = years[~np.isnan(years)]
    return UserState(
        n_ratings=float(it.size),
        n_pos=float(liked.size),
        mean_rating=float(rt.mean()),
        days_since_last=(now - ts[-1]) / SECONDS_PER_DAY,
        span_days=(ts[-1] - ts[0]) / SECONDS_PER_DAY,
        mean_log_pop=float(table.log_pop[liked].mean()) if liked.size else np.nan,
        mean_year=float(years.mean()) if years.size else np.nan,
        genre_share=share,
        recent=liked[-recent_n:] if recent_n > 0 else liked[:0],
    )


def user_states_from_history(
    history: pd.DataFrame, users: Sequence[int] | IntArray, table: ItemTable
) -> list[UserState]:
    """Offline path: `user_state` for each requested user, from their history rows."""
    cols = [ITEM_IDX, RATING, TIMESTAMP, IS_POSITIVE]
    groups = {
        u: g[cols].to_numpy() for u, g in history[[USER_IDX, *cols]].groupby(USER_IDX, sort=False)
    }
    empty = np.empty((0, 4))
    states = []
    for u in users:
        rows = groups.get(int(u), empty)
        states.append(user_state(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], table))
    return states


@dataclass(frozen=True)
class UserBlock:
    """User states stacked into arrays (row r = r-th user), plus demographic codes."""

    scalars: dict[str, FloatArray]
    genre_share: FloatArray  # [n, G]
    recent: list[IntArray]
    gender: IntArray
    age: IntArray
    occupation: IntArray

    @classmethod
    def stack(
        cls, states: Sequence[UserState], users: npt.ArrayLike, demographics: UserFeatures
    ) -> UserBlock:
        u = np.asarray(users, dtype=np.int64)
        return cls(
            scalars={
                c: np.array([getattr(s, c) for s in states], dtype=np.float64) for c in USER_SCALARS
            },
            genre_share=np.vstack([s.genre_share for s in states]),
            recent=[s.recent for s in states],
            gender=demographics.gender[u],
            age=demographics.age[u],
            occupation=demographics.occupation[u],
        )


def _embedding_similarity(
    block: UserBlock, rows: IntArray, items: IntArray, item_vectors: npt.NDArray[np.floating]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Cosine of each candidate to the user's recent liked items: max, mean, and last."""
    out = np.full((3, len(rows)), np.nan)
    order = np.argsort(rows, kind="stable")
    bounds = np.flatnonzero(np.diff(rows[order])) + 1
    for pairs in np.split(order, bounds):
        if pairs.size == 0:
            continue
        recent = block.recent[rows[pairs[0]]]
        if recent.size == 0:
            continue
        sims = item_vectors[items[pairs]] @ item_vectors[recent].T  # [n_pairs, n_recent]
        out[0, pairs], out[1, pairs], out[2, pairs] = sims.max(1), sims.mean(1), sims[:, -1]
    return out[0], out[1], out[2]


def compute_features(
    block: UserBlock,
    rows: npt.ArrayLike,
    items: npt.ArrayLike,
    table: ItemTable,
    item_vectors: npt.NDArray[np.floating],
    retrieval: Mapping[str, npt.ArrayLike],
) -> pd.DataFrame:
    """Feature frame for pairs (block row `rows[p]`, item `items[p]`)."""
    return pd.DataFrame(feature_columns(block, rows, items, table, item_vectors, retrieval))


def feature_matrix(columns: Mapping[str, npt.ArrayLike], names: Sequence[str]) -> FloatArray:
    """[n_pairs, len(names)] float64 matrix in the ranker's feature order."""
    return np.column_stack([np.asarray(columns[n], dtype=np.float64) for n in names])


def feature_columns(
    block: UserBlock,
    rows: npt.ArrayLike,
    items: npt.ArrayLike,
    table: ItemTable,
    item_vectors: npt.NDArray[np.floating],
    retrieval: Mapping[str, npt.ArrayLike],
) -> dict[str, npt.NDArray]:
    """THE feature implementation: one column per feature for pairs (block row `rows[p]`,
    item `items[p]`).

    `retrieval` holds per-pair retriever outputs (scores, ranks, number of sources); they
    are passed through unchanged so serving supplies them from the same retrievers.
    Item statistics stay as of the table's cutoff: they are batch features, refreshed with
    each bundle. The only time-dependent user feature (days since last interaction) is
    already measured from the decision time inside `user_state`.
    """
    rows = np.asarray(rows, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    f: dict[str, npt.ArrayLike] = {}
    for name, values in block.scalars.items():
        f[f"user_{name}"] = values[rows]
    f["user_gender"], f["user_age"] = block.gender[rows], block.age[rows]
    f["user_occupation"] = block.occupation[rows]
    for name in table.numeric.columns:
        f[name] = table.numeric[name].to_numpy()[items]
    genres = table.genres[items]
    for j, name in enumerate(table.genre_names):
        f[f"genre_{name}"] = genres[:, j]

    # cross features: how this item relates to this user's taste
    share = block.genre_share[rows] * genres
    f["x_genre_affinity_mean"] = share.sum(1) / np.maximum(genres.sum(1), 1.0)
    f["x_genre_affinity_max"] = share.max(1)
    f["x_year_gap"] = np.abs(table.year[items] - block.scalars["mean_year"][rows])
    f["x_pop_gap"] = table.log_pop[items] - block.scalars["mean_log_pop"][rows]
    sim_max, sim_mean, sim_last = _embedding_similarity(block, rows, items, item_vectors)
    f["x_sim_recent_max"], f["x_sim_recent_mean"], f["x_sim_last"] = sim_max, sim_mean, sim_last

    for name, values in retrieval.items():
        f[name] = np.asarray(values)
    return {k: np.asarray(v) for k, v in f.items()}
