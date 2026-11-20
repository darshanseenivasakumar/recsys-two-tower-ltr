"""Item-item k-nearest-neighbours with (shrunk) cosine similarity.

    sim(i, j)   = |U_i ∩ U_j| / (sqrt(|U_i|) * sqrt(|U_j|) + shrink)
    score(u, j) = sum over items i liked by u of sim(i, j), keeping only each item's
                  top-k most similar neighbours

U_i is the set of users who liked item i. `shrink` pulls similarities computed from very
few co-occurrences toward 0 (two niche movies both liked by a single user would
otherwise get cosine 1.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.models.base import Recommender, Scores, Users
from recsys.models.matrix import interaction_matrix


def topk_cosine_similarity(
    x: sp.csr_matrix, k: int, shrink: float = 0.0, block_size: int = 1024
) -> sp.csr_matrix:
    """Item-item cosine similarity of the columns of `x`, pruned to the top-k per row.

    Computed in blocks of rows so memory is O(block_size * n_items) instead of
    O(n_items^2). Matters for ML-25M (62k items: a dense float32 matrix would be 15 GB).
    Row i of the result holds item i's k most similar items; the diagonal is excluded.
    """
    n_items = x.shape[1]
    k = min(k, n_items - 1)
    xt = x.T.tocsr()  # items x users
    norms = np.sqrt(np.asarray(x.sum(axis=0)).ravel()).astype(np.float32)

    rows, cols, vals = [], [], []
    for start in range(0, n_items, block_size):
        stop = min(start + block_size, n_items)
        dots = (xt[start:stop] @ x).toarray().astype(np.float32)  # [block, n_items] co-counts
        sims = dots / (norms[start:stop, None] * norms[None, :] + shrink + 1e-8)
        sims[np.arange(stop - start), np.arange(start, stop)] = 0.0  # no self-similarity

        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        top_vals = np.take_along_axis(sims, top, axis=1)
        keep = top_vals > 0
        rows.append(np.repeat(np.arange(start, stop), k)[keep.ravel()])
        cols.append(top[keep])
        vals.append(top_vals[keep])

    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_items, n_items),
        dtype=np.float32,
    )


@dataclass
class ItemKNN(Recommender):
    k_neighbors: int = 100
    shrink: float = 0.0
    user_items_: sp.csr_matrix = field(init=False, repr=False)
    similarity_: sp.csr_matrix = field(init=False, repr=False)

    name = "item_knn"

    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> Self:
        self.user_items_ = interaction_matrix(history, n_users, n_items)
        self.similarity_ = topk_cosine_similarity(self.user_items_, self.k_neighbors, self.shrink)
        return self

    def score(self, users: Users) -> Scores:
        # [b, n_items] @ [n_items, n_items]: sum of neighbour similarities of liked items.
        return (self.user_items_[users] @ self.similarity_).toarray()
