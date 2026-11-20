"""Recent-history features: the user's last L liked items, as padded token arrays.

Tokens are item_idx + 1, with 0 as padding, newest item LAST:

    liked (oldest -> newest) = [7, 3, 9],  L = 5   ->   [0, 0, 8, 4, 10]

Two entry points produce this, and must agree exactly (tests enforce it):

* `encode_history(items, L)`   serving: from a user's raw recent-item list (e.g. Redis)
* `HistoryIndex.gather(...)`  training / offline eval: vectorized over many
                               (user, position) pairs, where position p means "the
                               user's first p liked items", i.e. strictly before example p.

Using position p (not the full history) for a training example whose target is the p-th
liked item is what prevents the model from seeing its own label in the input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.data.schema import IS_POSITIVE, ITEM_ID, ITEM_IDX, TIMESTAMP, USER_IDX

IntArray = npt.NDArray[np.int64]
PAD_TOKEN = 0


def encode_history(items: Sequence[int] | IntArray, max_len: int) -> IntArray:
    """Serving path: last `max_len` item_idx (oldest -> newest) -> left-padded tokens."""
    recent = np.asarray(items, dtype=np.int64)[-max_len:] if max_len > 0 else np.empty(0, np.int64)
    out = np.full(max_len, PAD_TOKEN, dtype=np.int64)
    if recent.size:
        out[max_len - recent.size :] = recent + 1
    return out


@dataclass(frozen=True)
class HistoryIndex:
    """Every user's liked items in time order, stored as one flat array + offsets (CSR-style).

    items[starts[u] : starts[u] + lengths[u]] is user u's liked items, oldest first.
    Memory is O(#positives), so no [n_examples, L] matrix is ever materialized up front.
    """

    items: IntArray
    starts: IntArray  # [n_users]
    lengths: IntArray  # [n_users]

    @classmethod
    def from_interactions(cls, history: pd.DataFrame, n_users: int) -> HistoryIndex:
        pos = history[history[IS_POSITIVE]].sort_values(
            [USER_IDX, TIMESTAMP, ITEM_ID], kind="stable"
        )
        lengths = np.bincount(pos[USER_IDX], minlength=n_users).astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        return cls(items=pos[ITEM_IDX].to_numpy(dtype=np.int64), starts=starts, lengths=lengths)

    def user_items(self, user: int) -> IntArray:
        return self.items[self.starts[user] : self.starts[user] + self.lengths[user]]

    def gather(self, users: IntArray, positions: IntArray, max_len: int) -> IntArray:
        """Tokens of each user's liked items strictly before `positions` (vectorized).

        Row r covers items at positions [positions[r] - max_len, positions[r]) of user r,
        left-padded exactly like `encode_history`.
        """
        users = np.asarray(users, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.int64)
        offsets = np.arange(-max_len, 0, dtype=np.int64)  # -L .. -1
        pos = positions[:, None] + offsets[None, :]  # [n, L] positions within the user
        valid = pos >= 0
        if self.items.size == 0:
            return np.full(pos.shape, PAD_TOKEN, dtype=np.int64)
        # Clamp before indexing: a trailing user with no items has starts[u] == len(items),
        # and numpy evaluates self.items[flat] before np.where applies the mask.
        flat = np.minimum(
            self.starts[users][:, None] + np.where(valid, pos, 0), self.items.size - 1
        )
        return np.where(valid, self.items[flat] + 1, PAD_TOKEN)

    def latest(self, users: IntArray, max_len: int) -> IntArray:
        """Each user's most recent `max_len` liked items (the eval/serving-time feature)."""
        users = np.asarray(users, dtype=np.int64)
        return self.gather(users, self.lengths[users], max_len)

    def training_examples(self) -> tuple[IntArray, IntArray, IntArray]:
        """(user, position, target_item) for every liked item, position = its index."""
        users = np.repeat(np.arange(len(self.lengths), dtype=np.int64), self.lengths)
        positions = np.arange(len(self.items), dtype=np.int64) - self.starts[users]
        return users, positions, self.items
