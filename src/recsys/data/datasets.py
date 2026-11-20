"""Dataset registry: one place that knows each MovieLens variant's URL and file format.

Every reader returns the same `RawTables` shape with canonical column names, so code
downstream of this module never needs to know which MovieLens variant it is handling.
Adding a dataset means writing one reader and registering one `DatasetSpec`.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data.schema import (
    AGE,
    GENDER,
    GENRES,
    ITEM_ID,
    OCCUPATION,
    RATING,
    TIMESTAMP,
    TITLE,
    USER_ID,
)


@dataclass(frozen=True)
class RawTables:
    """Raw tables in canonical column names, before any cleaning.

    interactions: user_id, item_id, rating, timestamp
    items:        item_id, title, genres (still the raw "A|B|C" string)
    users:        user_id [, gender, age, occupation]  (demographics only if the dataset has them)
    """

    interactions: pd.DataFrame
    items: pd.DataFrame
    users: pd.DataFrame


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    md5: str  # official checksum published by GroupLens alongside the archive
    extracted_dir: str  # top-level folder inside the zip
    reader: Callable[[Path], RawTables]
    has_user_features: bool


def _read_dat(path: Path, names: list[str]) -> pd.DataFrame:
    """Read ML-1M's '::'-separated, latin-1 encoded .dat files.

    A multi-character separator forces pandas' (slower) python engine; fine at 1M rows.
    QUOTE_NONE because some titles contain quote characters that are not CSV quoting.
    """
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=names,
        encoding="latin-1",
        quoting=csv.QUOTE_NONE,
        dtype=str,
    )


def read_ml_1m(root: Path) -> RawTables:
    ratings = _read_dat(root / "ratings.dat", [USER_ID, ITEM_ID, RATING, TIMESTAMP])
    ratings = ratings.astype(
        {USER_ID: "int32", ITEM_ID: "int32", RATING: "float32", TIMESTAMP: "int64"}
    )

    movies = _read_dat(root / "movies.dat", [ITEM_ID, TITLE, GENRES])
    movies[ITEM_ID] = movies[ITEM_ID].astype("int32")

    users = _read_dat(root / "users.dat", [USER_ID, GENDER, AGE, OCCUPATION, "zip_code"])
    users = users.drop(columns="zip_code").astype(
        {USER_ID: "int32", AGE: "int16", OCCUPATION: "int16"}
    )
    return RawTables(interactions=ratings, items=movies, users=users)


def read_ml_25m(root: Path) -> RawTables:
    ratings = pd.read_csv(
        root / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    ).rename(columns={"userId": USER_ID, "movieId": ITEM_ID})

    movies = pd.read_csv(root / "movies.csv", dtype={"movieId": "int32"}).rename(
        columns={"movieId": ITEM_ID}
    )
    # ML-25M has no users file: the user "table" is just the set of IDs seen in ratings.
    users = pd.DataFrame({USER_ID: np.sort(ratings[USER_ID].unique())})
    return RawTables(interactions=ratings, items=movies, users=users)


DATASETS: dict[str, DatasetSpec] = {
    "ml-1m": DatasetSpec(
        name="ml-1m",
        url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        md5="c4d9eecfca2ab87c1945afe126590906",
        extracted_dir="ml-1m",
        reader=read_ml_1m,
        has_user_features=True,
    ),
    "ml-25m": DatasetSpec(
        name="ml-25m",
        url="https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        md5="6b51fb2759a8657d3bfcbfc42b592ada",
        extracted_dir="ml-25m",
        reader=read_ml_25m,
        has_user_features=False,
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset {name!r}; available: {sorted(DATASETS)}") from None
