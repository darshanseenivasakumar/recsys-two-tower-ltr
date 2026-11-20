from __future__ import annotations

import pandas as pd
import pytest

from recsys.data.schema import ITEM_ID, TIMESTAMP, USER_ID
from recsys.data.split import (
    SplitResult,
    global_temporal_split,
    leave_last_n_split,
    summarize_split,
)
from tests.conftest import make_interactions

KEY = [USER_ID, ITEM_ID]


def _assert_partition(split: SplitResult, original: pd.DataFrame) -> None:
    """train/val/test are disjoint and together contain every original row."""
    parts = [split.train, split.val, split.test]
    combined = pd.concat(parts)
    assert len(combined) == len(original)
    assert not combined.duplicated(subset=KEY).any()
    assert set(map(tuple, combined[KEY].to_numpy())) == set(map(tuple, original[KEY].to_numpy()))


# ---------------------------------------------------------------- global temporal


def test_global_split_is_a_partition(toy_interactions: pd.DataFrame) -> None:
    _assert_partition(global_temporal_split(toy_interactions, 0.2, 0.2), toy_interactions)


def test_global_split_never_trains_on_the_future() -> None:
    df = make_interactions([(i % 7, i, 4.0, i) for i in range(100)])  # 7 users, ts 0..99
    split = global_temporal_split(df, val_frac=0.1, test_frac=0.2)
    assert split.train[TIMESTAMP].max() < split.val[TIMESTAMP].min()
    assert split.val[TIMESTAMP].max() < split.test[TIMESTAMP].min()
    assert split.meta["t_val"] == split.val[TIMESTAMP].min()
    assert split.meta["t_test"] == split.test[TIMESTAMP].min()
    # With 100 distinct, evenly spaced timestamps the sizes are exact.
    assert (len(split.train), len(split.val), len(split.test)) == (70, 10, 20)


def test_global_split_keeps_timestamp_ties_together() -> None:
    # Six events at ts=5: a cutoff landing there must not separate them.
    df = make_interactions([(1, i, 4.0, 5 if 2 <= i < 8 else i) for i in range(10)])
    split = global_temporal_split(df, val_frac=0.3, test_frac=0.3)
    sides = {name for name, part in split.parts().items() if (part[TIMESTAMP] == 5).any()}
    assert len(sides) == 1


# ---------------------------------------------------------------- leave last n
# These define the contract for leave_last_n_split (implemented by hand, see TODO(human)).


def test_leave_last_n_is_a_partition(toy_interactions: pd.DataFrame) -> None:
    _assert_partition(leave_last_n_split(toy_interactions, n_val=1, n_test=1), toy_interactions)


def test_leave_last_n_holds_out_latest_items(toy_interactions: pd.DataFrame) -> None:
    split = leave_last_n_split(toy_interactions, n_val=1, n_test=1)
    u1_test = split.test[split.test[USER_ID] == 1][ITEM_ID].tolist()
    u1_val = split.val[split.val[USER_ID] == 1][ITEM_ID].tolist()
    assert u1_test == [15]
    assert u1_val == [14]
    assert sorted(split.train[split.train[USER_ID] == 1][ITEM_ID]) == [11, 12, 13]


def test_leave_last_n_tie_break_is_by_item_id(toy_interactions: pd.DataFrame) -> None:
    # User 2: items 21 and 22 share ts=40, tie-break by item_id makes 22 the "latest".
    split = leave_last_n_split(toy_interactions, n_val=1, n_test=1)
    assert split.test[split.test[USER_ID] == 2][ITEM_ID].tolist() == [22]
    assert split.val[split.val[USER_ID] == 2][ITEM_ID].tolist() == [21]


def test_leave_last_n_is_independent_of_row_order(toy_interactions: pd.DataFrame) -> None:
    shuffled = toy_interactions.sample(frac=1.0, random_state=0)
    a = leave_last_n_split(toy_interactions, n_val=1, n_test=1)
    b = leave_last_n_split(shuffled, n_val=1, n_test=1)
    for part_a, part_b in zip(a.parts().values(), b.parts().values(), strict=True):
        assert sorted(map(tuple, part_a[KEY].to_numpy())) == sorted(
            map(tuple, part_b[KEY].to_numpy())
        )


@pytest.mark.parametrize(("n_val", "n_test"), [(1, 1), (0, 2), (2, 2)])
def test_leave_last_n_every_user_keeps_training_history(
    toy_interactions: pd.DataFrame, n_val: int, n_test: int
) -> None:
    split = leave_last_n_split(toy_interactions, n_val=n_val, n_test=n_test)
    assert set(split.train[USER_ID]) == set(toy_interactions[USER_ID])
    _assert_partition(split, toy_interactions)


# ---------------------------------------------------------------- summary


def test_summary_counts_cold_users_and_items() -> None:
    df = make_interactions(
        [
            (1, 10, 5.0, 1),
            (1, 11, 5.0, 2),
            (2, 10, 5.0, 3),
            (3, 12, 5.0, 9),  # user 3 and item 12 appear only in test -> both cold
        ]
    )
    split = global_temporal_split(df, val_frac=0.01, test_frac=0.3)
    test_stats = summarize_split(split)["test"]
    assert test_stats["cold_users"] == 1
    assert test_stats["cold_items"] == 1
    assert test_stats["positives_on_cold_items"] == 1
