"""Implicit-feedback matrix factorization via Alternating Least Squares (Hu, Koren, Volinsky 2008).

Objective over ALL user-item pairs (observed and unobserved):

    min  sum_{u,i} c_ui (p_ui - x_u . y_i)^2  +  lambda (sum_u |x_u|^2 + sum_i |y_i|^2)
    p_ui = 1 if u liked i else 0         (preference)
    c_ui = 1 + alpha * r_ui              (confidence; r_ui = 1 for our binary positives)

Unobserved pairs are weak negatives (p=0 with confidence 1), observed positives are strong
(confidence 1 + alpha). Fixing the item vectors makes the problem least-squares in the
user vectors (closed form), and vice versa, so it alternates. `implicit` does the heavy
lifting with conjugate-gradient steps, exploiting the sparsity of the matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from implicit.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits

from recsys.models.base import Recommender, Scores, Users
from recsys.models.matrix import interaction_matrix


@dataclass
class ALS(Recommender):
    factors: int = 64
    regularization: float = 0.01
    alpha: float = 10.0
    iterations: int = 15
    seed: int = 42
    user_factors_: npt.NDArray[np.float32] = field(init=False, repr=False)
    item_factors_: npt.NDArray[np.float32] = field(init=False, repr=False)

    name = "als"

    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> Self:
        user_items = interaction_matrix(history, n_users, n_items)
        # implicit parallelizes itself; a multi-threaded BLAS underneath oversubscribes cores.
        # The limit must cover construction too: that's where implicit checks the BLAS config.
        with threadpool_limits(limits=1, user_api="blas"):
            model = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                alpha=self.alpha,
                iterations=self.iterations,
                random_state=self.seed,
                use_gpu=False,
            )
            model.fit(user_items, show_progress=False)
        self.user_factors_ = np.asarray(model.user_factors)
        self.item_factors_ = np.asarray(model.item_factors)
        return self

    def score(self, users: Users) -> Scores:
        return self.user_factors_[users] @ self.item_factors_.T
