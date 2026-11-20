"""Static user and item features as integer arrays ready for embedding lookups.

Convention: index 0 always means "padding / unknown", so every vocabulary is shifted by +1.
That lets a missing value (ML-25M has no demographics, some titles have no year) use the
same code path as a present one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.data.schema import AGE, GENDER, GENRES, ITEM_IDX, OCCUPATION, USER_IDX, YEAR

IntArray = npt.NDArray[np.int64]

YEAR_BUCKET_START = 1900
YEAR_BUCKET_WIDTH = 5
N_YEAR_BUCKETS = 30  # 1900..2049 in 5-year buckets; earlier/later years are clamped


def year_bucket(years: pd.Series) -> IntArray:
    """Release year -> bucket id in 1..N_YEAR_BUCKETS, 0 for unknown year."""
    y = years.astype("Float64")
    bucket = ((y - YEAR_BUCKET_START) // YEAR_BUCKET_WIDTH).clip(0, N_YEAR_BUCKETS - 1) + 1
    return bucket.fillna(0).astype(np.int64).to_numpy()


def categorical_codes(values: pd.Series) -> tuple[IntArray, list[str]]:
    """Map values to 1..V in sorted order (0 = missing). Returns (codes, vocabulary)."""
    vocab = sorted(values.dropna().astype(str).unique())
    lookup = {v: i + 1 for i, v in enumerate(vocab)}
    codes = values.astype("string").map(lookup).fillna(0).astype(np.int64).to_numpy()
    return codes, vocab


@dataclass(frozen=True)
class ItemFeatures:
    """Per-item arrays indexed by item_idx."""

    genre_ids: IntArray  # [n_items, max_genres], 0-padded
    year_bucket: IntArray  # [n_items]
    genre_vocab: list[str]

    @property
    def n_genres(self) -> int:
        return len(self.genre_vocab) + 1  # + padding

    @classmethod
    def from_items(cls, items: pd.DataFrame) -> ItemFeatures:
        items = items.sort_values(ITEM_IDX)
        if not np.array_equal(items[ITEM_IDX].to_numpy(), np.arange(len(items))):
            raise ValueError("items must have contiguous item_idx 0..n-1")
        vocab = sorted({g for genres in items[GENRES] for g in genres})
        lookup = {g: i + 1 for i, g in enumerate(vocab)}
        max_genres = max(1, max(len(g) for g in items[GENRES]))
        genre_ids = np.zeros((len(items), max_genres), dtype=np.int64)
        for row, genres in enumerate(items[GENRES]):
            genre_ids[row, : len(genres)] = [lookup[g] for g in genres]
        return cls(genre_ids=genre_ids, year_bucket=year_bucket(items[YEAR]), genre_vocab=vocab)


@dataclass(frozen=True)
class UserFeatures:
    """Per-user demographic codes indexed by user_idx (all zeros if the dataset has none)."""

    gender: IntArray
    age: IntArray
    occupation: IntArray
    vocab_sizes: dict[str, int]

    @classmethod
    def from_users(cls, users: pd.DataFrame) -> UserFeatures:
        users = users.sort_values(USER_IDX)
        if not np.array_equal(users[USER_IDX].to_numpy(), np.arange(len(users))):
            raise ValueError("users must have contiguous user_idx 0..n-1")
        arrays: dict[str, IntArray] = {}
        sizes: dict[str, int] = {}
        for col in (GENDER, AGE, OCCUPATION):
            if col in users:
                arrays[col], vocab = categorical_codes(users[col])
                sizes[col] = len(vocab) + 1
            else:
                arrays[col] = np.zeros(len(users), dtype=np.int64)
                sizes[col] = 1
        return cls(arrays[GENDER], arrays[AGE], arrays[OCCUPATION], sizes)
