"""Two-tower retrieval model trained with in-batch sampled softmax and logQ correction.

    user tower:  [user-ID emb, mean(recent liked item embs), gender, age, occupation] -> MLP -> u
    item tower:  [item-ID emb, mean(genre embs), release-year bucket]                 -> MLP -> v
    score(u, v) = <u, v> / temperature,   with |u| = |v| = 1  (so <u, v> = cosine similarity)

Training (per batch of B positive (user, item) pairs):

    logits[i, j] = <u_i, v_j> / T  -  alpha * log q_j  (logQ correction, training only)
    loss         = mean_i  -log softmax(logits[i, :])[i]

Each row's own item is its positive; the other B-1 items in the batch are its negatives.
They are other users' positives, so item j appears as a negative with probability ~ q_j
(its share of all positives). Without correction, popular items are over-penalized and
the model under-recommends them relative to a full softmax; subtracting log q_j makes the
sampled softmax an unbiased-in-expectation estimate of the full one (Yi et al., RecSys 2019).
At serving time the plain <u, v> is used: no correction, no temperature needed for ranking.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812 (conventional alias)
from torch import nn

from recsys.features.encoders import N_YEAR_BUCKETS, ItemFeatures, UserFeatures
from recsys.features.history import HistoryIndex
from recsys.models.base import Recommender, Scores, Users

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TwoTowerConfig:
    id_dim: int = 64
    side_dim: int = 16
    hidden_dim: int = 128
    out_dim: int = 64
    max_history: int = 50
    temperature: float = 0.05
    logq_correction: bool = True
    logq_alpha: float = 1.0  # strength of the correction: logits - alpha * log q (1 = full)
    use_user_id: bool = True  # False: user tower = history + demographics only
    user_id_dropout: float = 0.1
    dropout: float = 0.0  # on each tower's MLP input
    weight_decay: float = 0.0  # AdamW (decoupled) weight decay on all parameters
    lr: float = 1e-3
    batch_size: int = 1024
    epochs: int = 10
    seed: int = 42
    device: str = "cpu"


def masked_mean(embeddings: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Mean over the sequence axis, ignoring padding tokens (0). All-padding rows -> zeros."""
    mask = (tokens != 0).unsqueeze(-1).to(embeddings.dtype)  # [b, L, 1]
    return (embeddings * mask).sum(1) / mask.sum(1).clamp(min=1.0)


def _mlp(in_dim: int, hidden: int, out: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out)
    )


class TwoTowerNet(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        item_feats: ItemFeatures,
        user_feats: UserFeatures,
        cfg: TwoTowerConfig,
    ) -> None:
        super().__init__()
        d, s = cfg.id_dim, cfg.side_dim
        # Shared by the item tower AND the user's history: token = item_idx + 1, 0 = pad.
        self.item_id = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.genre = nn.Embedding(item_feats.n_genres, s, padding_idx=0)
        self.year = nn.Embedding(N_YEAR_BUCKETS + 1, s)  # 0 = unknown year, learned
        self.user_id = nn.Embedding(n_users, d)
        self.gender = nn.Embedding(user_feats.vocab_sizes["gender"], s)
        self.age = nn.Embedding(user_feats.vocab_sizes["age"], s)
        self.occupation = nn.Embedding(user_feats.vocab_sizes["occupation"], s)
        self.item_mlp = _mlp(d + 2 * s, cfg.hidden_dim, cfg.out_dim, cfg.dropout)
        self.user_mlp = _mlp(2 * d + 3 * s, cfg.hidden_dim, cfg.out_dim, cfg.dropout)
        for emb in (self.item_id, self.user_id):
            nn.init.normal_(emb.weight, std=0.05)

    def item_vectors(
        self, item_tokens: torch.Tensor, genre_ids: torch.Tensor, year: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat(
            [
                self.item_id(item_tokens),
                masked_mean(self.genre(genre_ids), genre_ids),
                self.year(year),
            ],
            dim=-1,
        )
        return F.normalize(self.item_mlp(x), dim=-1)

    def user_vectors(
        self,
        users: torch.Tensor,
        use_id: torch.Tensor,
        history: torch.Tensor,
        gender: torch.Tensor,
        age: torch.Tensor,
        occupation: torch.Tensor,
    ) -> torch.Tensor:
        uid = self.user_id(users) * use_id.unsqueeze(-1).to(torch.float32)
        hist = masked_mean(self.item_id(history), history)
        x = torch.cat(
            [uid, hist, self.gender(gender), self.age(age), self.occupation(occupation)], dim=-1
        )
        return F.normalize(self.user_mlp(x), dim=-1)


F32 = npt.NDArray[np.float32]


@dataclass(frozen=True)
class NumpyUserTower:
    """`TwoTowerNet.user_vectors` in eval mode, as plain numpy, for serving one user at a time.

    Same math (embedding lookups, masked mean, Linear -> ReLU -> Linear, L2 normalize; dropout
    is the identity in eval), same float32 weights. For a single request torch spends most of
    its time dispatching a dozen tiny ops; here each is one numpy call. Results match torch
    to float32 rounding (summation order differs), checked in tests/test_two_tower.py.
    """

    user_id: F32
    item_id: F32  # row 0 is the padding token (zeros)
    gender: F32
    age: F32
    occupation: F32
    w1t: F32  # [in, hidden], transposed once so the request does x @ w1t
    b1: F32
    w2t: F32  # [hidden, out]
    b2: F32

    @classmethod
    def from_net(cls, net: TwoTowerNet) -> NumpyUserTower:
        s = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
        # _mlp = Sequential(Dropout, Linear, ReLU, Linear): Linear layers are indices 1 and 3
        return cls(
            user_id=s["user_id.weight"],
            item_id=s["item_id.weight"],
            gender=s["gender.weight"],
            age=s["age.weight"],
            occupation=s["occupation.weight"],
            w1t=np.ascontiguousarray(s["user_mlp.1.weight"].T),
            b1=s["user_mlp.1.bias"],
            w2t=np.ascontiguousarray(s["user_mlp.3.weight"].T),
            b2=s["user_mlp.3.bias"],
        )

    def __call__(
        self,
        users: npt.NDArray[np.int64],
        use_id: npt.NDArray[np.bool_],
        history: npt.NDArray[np.int64],
        gender: npt.NDArray[np.int64],
        age: npt.NDArray[np.int64],
        occupation: npt.NDArray[np.int64],
    ) -> F32:
        uid = self.user_id[users] * use_id[:, None].astype(np.float32)
        mask = (history != 0)[..., None].astype(np.float32)  # [b, L, 1]
        summed = (self.item_id[history] * mask).sum(axis=1)
        hist = summed / np.maximum(mask.sum(axis=1), np.float32(1.0))
        x = np.concatenate(
            [uid, hist, self.gender[gender], self.age[age], self.occupation[occupation]], axis=-1
        )
        h = np.maximum(x @ self.w1t + self.b1, np.float32(0.0))
        y = h @ self.w2t + self.b2
        norm = np.sqrt((y * y).sum(axis=-1, keepdims=True))
        return (y / np.maximum(norm, np.float32(1e-12))).astype(np.float32, copy=False)


def in_batch_softmax_loss(
    user_vecs: torch.Tensor,
    item_vecs: torch.Tensor,
    item_ids: torch.Tensor,
    temperature: float,
    log_q: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sampled softmax over in-batch negatives, with optional logQ correction.

    user_vecs, item_vecs: [B, d] L2-normalized; row i of each is a positive pair.
    item_ids: [B] so that duplicate items in the batch aren't used as negatives for a
              row whose positive IS that item ("accidental hits").
    log_q:    [B] log sampling probability of each in-batch item, or None for no correction.
    """
    logits = user_vecs @ item_vecs.T / temperature  # [B, B]
    if log_q is not None:
        logits = logits - log_q.unsqueeze(0)  # column j is penalized by log q_j
    batch = logits.shape[0]
    same_item = item_ids.unsqueeze(0) == item_ids.unsqueeze(1)
    accidental = same_item & ~torch.eye(batch, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(accidental, float("-inf"))
    labels = torch.arange(batch, device=logits.device)
    return F.cross_entropy(logits, labels)


EpochCallback = Callable[[int, "TwoTowerRecommender"], dict[str, float]]


@dataclass
class TwoTowerRecommender(Recommender):
    """Fits the two-tower net on positives from `history`; scores by exact dot product."""

    config: TwoTowerConfig
    item_features: ItemFeatures
    user_features: UserFeatures
    epoch_metrics_: list[dict[str, float]] = field(init=False, default_factory=list, repr=False)
    net_: TwoTowerNet = field(init=False, repr=False)
    history_: HistoryIndex = field(init=False, repr=False)
    known_users_: npt.NDArray[np.bool_] = field(init=False, repr=False)
    item_vectors_: npt.NDArray[np.float32] = field(init=False, repr=False)
    _numpy_tower: NumpyUserTower | None = field(init=False, default=None, repr=False)

    name = "two_tower"

    def params(self) -> dict[str, Any]:
        return asdict(self.config)

    @classmethod
    def from_trained(
        cls,
        config: TwoTowerConfig,
        item_features: ItemFeatures,
        user_features: UserFeatures,
        state_dict: dict[str, torch.Tensor],
        known_users: npt.NDArray[np.bool_],
        item_vectors: npt.NDArray[np.float32],
    ) -> TwoTowerRecommender:
        """Rebuild a fitted model from exported weights (serving: no training data needed)."""
        model = cls(config, item_features, user_features)
        n_users, n_items = len(known_users), len(item_vectors)
        model.net_ = TwoTowerNet(n_users, n_items, item_features, user_features, config)
        model.net_.load_state_dict(state_dict)
        model.net_.to(config.device).eval()
        model.known_users_ = np.asarray(known_users, dtype=bool)
        model.item_vectors_ = np.asarray(item_vectors, dtype=np.float32)
        return model

    def _tensor(self, array: npt.ArrayLike) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array), device=self.config.device)

    def fit(
        self,
        history: pd.DataFrame,
        n_users: int,
        n_items: int,
        on_epoch_end: EpochCallback | None = None,
    ) -> Self:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)

        self.history_ = HistoryIndex.from_interactions(history, n_users)
        self.known_users_ = self.history_.lengths > 0
        users, positions, targets = self.history_.training_examples()
        counts = np.bincount(targets, minlength=n_items)
        log_q_all = self._tensor(np.log(np.maximum(counts, 1) / counts.sum()).astype(np.float32))

        self.net_ = TwoTowerNet(n_users, n_items, self.item_features, self.user_features, cfg)
        self._numpy_tower = None  # any earlier export belongs to the previous weights
        self.net_.to(cfg.device)
        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        uf, itf = self.user_features, self.item_features
        self.epoch_metrics_ = []

        for epoch in range(cfg.epochs):
            self.net_.train()
            order = rng.permutation(len(targets))
            total, n_batches = 0.0, 0
            for start in range(0, len(order), cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                u, p, i = users[idx], positions[idx], targets[idx]
                hist = self.history_.gather(u, p, cfg.max_history)  # strictly before target
                use_id = (rng.random(len(idx)) >= cfg.user_id_dropout) & cfg.use_user_id

                user_vecs = self.net_.user_vectors(
                    self._tensor(u),
                    self._tensor(use_id),
                    self._tensor(hist),
                    self._tensor(uf.gender[u]),
                    self._tensor(uf.age[u]),
                    self._tensor(uf.occupation[u]),
                )
                item_t = self._tensor(i)
                item_vecs = self.net_.item_vectors(
                    item_t + 1, self._tensor(itf.genre_ids[i]), self._tensor(itf.year_bucket[i])
                )
                log_q = cfg.logq_alpha * log_q_all[item_t] if cfg.logq_correction else None
                loss = in_batch_softmax_loss(user_vecs, item_vecs, item_t, cfg.temperature, log_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                n_batches += 1

            self.item_vectors_ = self.item_embeddings()
            metrics = {"train_loss": total / n_batches}
            if on_epoch_end is not None:
                metrics |= on_epoch_end(epoch, self)
            self.epoch_metrics_.append(metrics)
            logger.info(
                "epoch %2d %s", epoch + 1, " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )
        return self

    @torch.no_grad()
    def item_embeddings(self, batch_size: int = 8192) -> npt.NDArray[np.float32]:
        """[n_items, out_dim] L2-normalized item vectors (what goes into the FAISS index)."""
        self.net_.eval()
        itf = self.item_features
        n_items = len(itf.year_bucket)
        chunks = []
        for start in range(0, n_items, batch_size):
            i = np.arange(start, min(start + batch_size, n_items))
            chunks.append(
                self.net_.item_vectors(
                    self._tensor(i + 1),
                    self._tensor(itf.genre_ids[i]),
                    self._tensor(itf.year_bucket[i]),
                )
                .cpu()
                .numpy()
            )
        return np.vstack(chunks)

    @torch.no_grad()
    def user_embeddings_from_history(
        self, users: Users, history_tokens: npt.NDArray[np.int64], use_id: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.float32]:
        """User vectors from explicit history tokens: the single path for eval AND serving."""
        self.net_.eval()
        uf = self.user_features
        return (
            self.net_.user_vectors(
                self._tensor(users),
                self._tensor(use_id),
                self._tensor(history_tokens),
                self._tensor(uf.gender[users]),
                self._tensor(uf.age[users]),
                self._tensor(uf.occupation[users]),
            )
            .cpu()
            .numpy()
        )

    def user_embeddings_numpy(
        self, users: Users, history_tokens: npt.NDArray[np.int64], use_id: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.float32]:
        """`user_embeddings_from_history` without torch (serving). The weights are exported
        once, on first use; call it only on a fitted model whose weights no longer change."""
        if self._numpy_tower is None:
            self._numpy_tower = NumpyUserTower.from_net(self.net_)
        users = np.asarray(users, dtype=np.int64)
        uf = self.user_features
        return self._numpy_tower(
            users,
            np.asarray(use_id, dtype=bool),
            np.asarray(history_tokens, dtype=np.int64),
            np.asarray(uf.gender[users], dtype=np.int64),
            np.asarray(uf.age[users], dtype=np.int64),
            np.asarray(uf.occupation[users], dtype=np.int64),
        )

    def user_embeddings(self, users: Users) -> npt.NDArray[np.float32]:
        """Vectors for users from the history this model was fit on (offline eval)."""
        users = np.asarray(users, dtype=np.int64)
        tokens = self.history_.latest(users, self.config.max_history)
        # Users never seen in training get no ID embedding (it would be untrained noise).
        use_id = self.known_users_[users] & self.config.use_user_id
        return self.user_embeddings_from_history(users, tokens, use_id)

    def score(self, users: Users) -> Scores:
        return self.user_embeddings(users) @ self.item_vectors_.T
