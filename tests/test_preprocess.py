from __future__ import annotations

import pandas as pd
import pytest

from recsys.data.datasets import RawTables
from recsys.data.preprocess import (
    deduplicate_interactions,
    parse_genres,
    parse_year,
    preprocess,
)
from recsys.data.schema import (
    GENRES,
    IS_POSITIVE,
    ITEM_ID,
    ITEM_IDX,
    RATING,
    TIMESTAMP,
    TITLE,
    USER_ID,
    USER_IDX,
    YEAR,
)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Toy Story (1995)", 1995),
        ("City of Lost Children, The (Cité des enfants perdus, La) (1995)", 1995),
        ("Babylon 5 ", None),
        ("Some Show (2006–2007)", None),
        ("Trailing Space (2001) ", 2001),
    ],
)
def test_parse_year(title: str, expected: int | None) -> None:
    assert parse_year(title) == expected


def test_parse_genres() -> None:
    assert parse_genres("Animation|Children's|Comedy") == ["Animation", "Children's", "Comedy"]
    assert parse_genres("(no genres listed)") == []
    assert parse_genres(float("nan")) == []


def test_deduplicate_keeps_latest_rating() -> None:
    df = pd.DataFrame(
        {USER_ID: [1, 1, 2], ITEM_ID: [7, 7, 7], RATING: [2.0, 5.0, 3.0], TIMESTAMP: [100, 200, 50]}
    )
    out = deduplicate_interactions(df)
    assert len(out) == 2
    assert out.loc[out[USER_ID] == 1, RATING].item() == 5.0


def _raw() -> RawTables:
    interactions = pd.DataFrame(
        {
            USER_ID: [42, 7, 42, 99],
            ITEM_ID: [300, 100, 100, 555],  # item 555 is not in the catalog
            RATING: [4.0, 3.5, 5.0, 4.0],
            TIMESTAMP: [30, 10, 20, 5],
        }
    )
    items = pd.DataFrame(
        {ITEM_ID: [300, 100, 200], TITLE: ["C (2000)", "A (1990)", "B"], GENRES: ["X|Y", "X", "Z"]}
    )
    users = pd.DataFrame({USER_ID: [42, 7, 99]})
    return RawTables(interactions=interactions, items=items, users=users)


def test_preprocess_builds_contiguous_indices_and_labels() -> None:
    out = preprocess(_raw(), positive_threshold=4.0)

    # Indices are dense 0..N-1, assigned in raw-ID order, and cover the full catalog.
    assert out.items[ITEM_ID].tolist() == [100, 200, 300]
    assert out.items[ITEM_IDX].tolist() == [0, 1, 2]
    assert out.users[USER_IDX].tolist() == [0, 1, 2]
    assert out.items[YEAR].isna().tolist() == [False, True, False]
    assert out.items[GENRES].tolist() == [["X"], ["Z"], ["X", "Y"]]

    inter = out.interactions
    assert 555 not in set(inter[ITEM_ID])  # unknown item dropped
    assert inter[TIMESTAMP].is_monotonic_increasing  # global chronological order
    assert inter[IS_POSITIVE].tolist() == [False, True, True]  # 3.5, 5.0, 4.0
    # idx columns agree with the lookup tables
    lookup = dict(zip(out.items[ITEM_ID], out.items[ITEM_IDX], strict=True))
    assert all(lookup[i] == j for i, j in zip(inter[ITEM_ID], inter[ITEM_IDX], strict=True))
