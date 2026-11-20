"""Shared fixtures: small hand-made interaction logs whose correct splits we can reason about."""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.data.schema import IS_POSITIVE, ITEM_ID, RATING, TIMESTAMP, USER_ID


def make_interactions(rows: list[tuple[int, int, float, int]]) -> pd.DataFrame:
    """rows: (user_id, item_id, rating, timestamp)."""
    df = pd.DataFrame(rows, columns=[USER_ID, ITEM_ID, RATING, TIMESTAMP])
    df[IS_POSITIVE] = df[RATING] >= 4.0
    return df


@pytest.fixture
def toy_interactions() -> pd.DataFrame:
    """Three users with different history lengths, including a timestamp tie.

    user 1: 5 interactions, times 10..50
    user 2: 3 interactions, items 21 and 22 share timestamp 40 (tie)
    user 3: 1 interaction  (too short to hold anything out)
    """
    return make_interactions(
        [
            (1, 11, 5.0, 10),
            (1, 12, 3.0, 20),
            (1, 13, 4.0, 30),
            (1, 14, 2.0, 40),
            (1, 15, 5.0, 50),
            (2, 20, 4.0, 15),
            (2, 22, 5.0, 40),
            (2, 21, 1.0, 40),
            (3, 30, 4.0, 25),
        ]
    )
