"""Sparse user-item matrix construction shared by the matrix-based models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.data.schema import IS_POSITIVE, ITEM_IDX, USER_IDX


def interaction_matrix(
    history: pd.DataFrame, n_users: int, n_items: int, positives_only: bool = True
) -> sp.csr_matrix:
    """Binary users x items CSR matrix (1 = interaction, implicitly 0 = unobserved).

    The shape comes from the catalog sizes, not from the data, so user_idx / item_idx
    line up with every other model and with the evaluation code even when some users or
    items have no interactions in this history.
    """
    df = history[history[IS_POSITIVE]] if positives_only else history
    data = np.ones(len(df), dtype=np.float32)
    matrix = sp.csr_matrix(
        (data, (df[USER_IDX].to_numpy(), df[ITEM_IDX].to_numpy())), shape=(n_users, n_items)
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0  # binary even if a pair somehow appears twice
    return matrix
