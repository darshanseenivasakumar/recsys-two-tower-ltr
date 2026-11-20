"""Matrix factorization trained with Bayesian Personalized Ranking (Rendle et al. 2009), in PyTorch.

Model:  x_ui = p_u . q_i + b_i                      (user/item embeddings + item bias)
Loss:   for a liked item i and a sampled item j the user hasn't liked,

            L = -log sigmoid(x_ui - x_uj) + reg * (|p_u|^2 + |q_i|^2 + |q_j|^2)

BPR optimizes the *order* (i above j), not the absolute score, which is what top-K
ranking needs. It's the pairwise ancestor of the two-tower model's sampled softmax in
Phase 3: same "positive vs. sampled negatives" idea, one negative instead of a batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn

from recsys.data.schema import IS_POSITIVE, ITEM_IDX, USER_IDX
from recsys.models.base import Recommender, Scores, Users

logger = logging.getLogger(__name__)


class _MF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int) -> None:
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return (self.user(u) * self.item(i)).sum(-1) + self.item_bias(i).squeeze(-1)


def sample_negatives(
    users: npt.NDArray[np.int64],
    positive_keys: npt.NDArray[np.int64],
    n_items: int,
    rng: np.random.Generator,
    max_tries: int = 5,
) -> npt.NDArray[np.int64]:
    """Uniformly sample one item per user that the user has NOT liked.

    positive_keys: sorted array of user * n_items + item for every positive pair, so a
    membership test is a vectorized binary search instead of a Python set lookup.
    Resamples collisions a few times; any left after that (vanishingly rare) are kept.
    """
    neg = rng.integers(0, n_items, size=len(users))
    for _ in range(max_tries):
        keys = users * n_items + neg
        pos = np.searchsorted(positive_keys, keys).clip(max=len(positive_keys) - 1)
        clash = positive_keys[pos] == keys
        if not clash.any():
            break
        neg[clash] = rng.integers(0, n_items, size=int(clash.sum()))
    return neg


@dataclass
class BPRMF(Recommender):
    dim: int = 64
    lr: float = 1e-3
    reg: float = 1e-5
    epochs: int = 30
    batch_size: int = 4096
    seed: int = 42
    device: str = "cpu"
    loss_history_: list[float] = field(init=False, default_factory=list, repr=False)
    model_: _MF = field(init=False, repr=False)

    name = "bpr"

    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> Self:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        pos = history[history[IS_POSITIVE]]
        users = pos[USER_IDX].to_numpy(dtype=np.int64)
        items = pos[ITEM_IDX].to_numpy(dtype=np.int64)
        positive_keys = np.sort(users * n_items + items)

        self.model_ = _MF(n_users, n_items, self.dim).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        self.loss_history_ = []

        for epoch in range(self.epochs):
            order = rng.permutation(len(users))
            total, n_batches = 0.0, 0
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                u_np = users[idx]
                j_np = sample_negatives(u_np, positive_keys, n_items, rng)
                u, i, j = (torch.from_numpy(a).to(self.device) for a in (u_np, items[idx], j_np))

                diff = self.model_(u, i) - self.model_(u, j)
                l2 = (
                    self.model_.user(u).pow(2).sum(-1)
                    + self.model_.item(i).pow(2).sum(-1)
                    + self.model_.item(j).pow(2).sum(-1)
                )
                loss = -nn.functional.logsigmoid(diff).mean() + self.reg * l2.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                n_batches += 1
            self.loss_history_.append(total / n_batches)
            logger.debug("epoch %d loss %.4f", epoch, self.loss_history_[-1])
        return self

    @torch.no_grad()
    def score(self, users: Users) -> Scores:
        self.model_.eval()
        u = torch.from_numpy(np.asarray(users, dtype=np.int64)).to(self.device)
        scores = self.model_.user(u) @ self.model_.item.weight.T + self.model_.item_bias.weight.T
        return scores.cpu().numpy()
