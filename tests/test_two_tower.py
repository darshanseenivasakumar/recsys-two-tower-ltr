"""Two-tower: feature parity (no skew), no label leakage, loss math, planted recovery, FAISS."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from recsys.data.schema import (
    AGE,
    GENDER,
    GENRES,
    IS_POSITIVE,
    ITEM_ID,
    ITEM_IDX,
    OCCUPATION,
    TIMESTAMP,
    USER_IDX,
    YEAR,
)
from recsys.features.encoders import ItemFeatures, UserFeatures, year_bucket
from recsys.features.history import PAD_TOKEN, HistoryIndex, encode_history
from recsys.models.two_tower import (
    TwoTowerConfig,
    TwoTowerRecommender,
    in_batch_softmax_loss,
    masked_mean,
)
from recsys.retrieval.ann import build_index, overlap_at_k, search_excluding_seen

# ------------------------------------------------------------------ history features


def _interactions(rows: list[tuple[int, int, bool, int]]) -> pd.DataFrame:
    """rows: (user_idx, item_idx, is_positive, timestamp); item_id mirrors item_idx."""
    df = pd.DataFrame(rows, columns=[USER_IDX, ITEM_IDX, IS_POSITIVE, TIMESTAMP])
    df[ITEM_ID] = df[ITEM_IDX]
    return df


def test_encode_history_left_pads_newest_last() -> None:
    assert encode_history([7, 3, 9], 5).tolist() == [0, 0, 8, 4, 10]
    assert encode_history([1, 2, 3, 4], 2).tolist() == [4, 5]  # keeps the most recent
    assert encode_history([], 3).tolist() == [PAD_TOKEN] * 3


def test_history_index_orders_by_time_and_ignores_negatives() -> None:
    hist = _interactions([(0, 5, True, 30), (0, 2, True, 10), (0, 9, False, 20), (2, 1, True, 5)])
    index = HistoryIndex.from_interactions(hist, n_users=4)
    assert index.user_items(0).tolist() == [2, 5]  # time order; the negative (9) is excluded
    assert index.lengths.tolist() == [2, 0, 1, 0]
    # trailing users with no items must not index out of bounds
    assert index.latest(np.array([3, 1]), 2).tolist() == [[0, 0], [0, 0]]


def test_training_and_serving_paths_produce_identical_features() -> None:
    """The training-serving skew test: same user, same cutoff -> byte-identical input."""
    rng = np.random.default_rng(0)
    rows = [(u, int(i), True, t) for u in range(6) for t, i in enumerate(rng.integers(0, 50, 12))]
    rows = list({(u, i): (u, i, p, t) for u, i, p, t in rows}.values())  # one row per pair
    index = HistoryIndex.from_interactions(_interactions(rows), n_users=6)
    max_len = 5
    for user in range(6):
        items = index.user_items(user)
        for p in range(len(items) + 1):
            training = index.gather(np.array([user]), np.array([p]), max_len)[0]
            serving = encode_history(items[:p], max_len)  # what the API would build from Redis
            np.testing.assert_array_equal(training, serving)


def test_training_examples_never_contain_their_own_target() -> None:
    hist = _interactions([(0, i, True, i) for i in range(8)])
    index = HistoryIndex.from_interactions(hist, n_users=1)
    users, positions, targets = index.training_examples()
    tokens = index.gather(users, positions, max_len=10)
    for row, target in enumerate(targets):
        seen = tokens[row][tokens[row] != PAD_TOKEN] - 1
        assert target not in seen  # no label leakage
        assert (seen < target).all()  # only items liked earlier (here item == time)


# ------------------------------------------------------------------ static features


def test_year_bucket() -> None:
    years = pd.Series([1995, None, 1850, 2100], dtype="Int16")
    # 1995 -> (95 // 5) + 1 = 20; unknown -> 0; clamped at both ends
    assert year_bucket(years).tolist() == [20, 0, 1, 30]


def test_item_and_user_features() -> None:
    items = pd.DataFrame(
        {
            ITEM_IDX: [1, 0],
            GENRES: [["Drama"], ["Comedy", "Drama"]],
            YEAR: pd.Series([2000, None], dtype="Int16"),
        }
    )
    feats = ItemFeatures.from_items(items)
    assert feats.genre_vocab == ["Comedy", "Drama"]
    assert feats.genre_ids.tolist() == [[1, 2], [2, 0]]  # row = item_idx, 0-padded
    assert feats.n_genres == 3

    users = pd.DataFrame({USER_IDX: [0, 1]})  # ML-25M style: no demographics
    uf = UserFeatures.from_users(users)
    assert uf.gender.tolist() == [0, 0] and uf.vocab_sizes["age"] == 1


# ------------------------------------------------------------------ loss


def test_masked_mean_ignores_padding() -> None:
    emb = torch.tensor([[[1.0, 1.0], [3.0, 3.0]], [[5.0, 5.0], [7.0, 7.0]]])
    tokens = torch.tensor([[0, 4], [0, 0]])
    assert masked_mean(emb, tokens).tolist() == [[3.0, 3.0], [0.0, 0.0]]


def test_in_batch_loss_hand_computed() -> None:
    eye = torch.eye(2)
    # logits = [[1, 0], [0, 1]] -> each row: -log(e / (e + 1)) = log(1 + e^-1)
    loss = in_batch_softmax_loss(eye, eye, torch.tensor([0, 1]), temperature=1.0)
    assert loss.item() == pytest.approx(math.log(1 + math.exp(-1)))


def test_logq_correction_hand_computed() -> None:
    eye = torch.eye(2)
    q = torch.tensor([0.8, 0.2])  # item 0 popular, item 1 rare
    loss = in_batch_softmax_loss(eye, eye, torch.tensor([0, 1]), 1.0, log_q=q.log())
    # row 0 logits: [1 - log .8, 0 - log .2];  row 1: [0 - log .8, 1 - log .2]
    row0 = -torch.log_softmax(torch.tensor([1 - math.log(0.8), -math.log(0.2)]), 0)[0]
    row1 = -torch.log_softmax(torch.tensor([-math.log(0.8), 1 - math.log(0.2)]), 0)[1]
    assert loss.item() == pytest.approx(((row0 + row1) / 2).item(), rel=1e-6)
    # Equal q for every item is a constant shift per row -> no effect on the loss.
    uniform = in_batch_softmax_loss(
        eye, eye, torch.tensor([0, 1]), 1.0, log_q=torch.log(torch.tensor([0.5, 0.5]))
    )
    plain = in_batch_softmax_loss(eye, eye, torch.tensor([0, 1]), 1.0)
    assert uniform.item() == pytest.approx(plain.item())


def test_accidental_hits_are_masked() -> None:
    u = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    ids = torch.tensor([7, 8, 7])  # rows 0 and 2 share item 7
    loss = in_batch_softmax_loss(u, v, ids, temperature=1.0)
    logits = u @ v.T
    logits[0, 2] = logits[2, 0] = float("-inf")  # item 7 isn't a negative for its own rows
    expected = torch.nn.functional.cross_entropy(logits, torch.arange(3))
    assert loss.item() == pytest.approx(expected.item())


# ------------------------------------------------------------------ end to end


def test_two_tower_recovers_planted_groups() -> None:
    """Users 0-19 like items 0-4 (genre A), users 20-39 like 5-9 (genre B); one held out each."""
    rows, groups = [], []
    for u in range(40):
        group = set(range(5)) if u < 20 else set(range(5, 10))
        held = min(group) + u % 5
        # Watch order must be random. If every user watched in item-id order, the history
        # tower correctly learns "next item = next id" and predicts the first item of the
        # OTHER group after a user's last one (hit rate 0.20, loss 0.26 in that version).
        order = np.random.default_rng(u).permutation(sorted(group - {held}))
        rows += [(u, int(i), True, t) for t, i in enumerate(order)]
        groups.append(group)
    hist = _interactions(rows)
    items = ItemFeatures.from_items(
        pd.DataFrame(
            {
                ITEM_IDX: range(10),
                GENRES: [["A"]] * 5 + [["B"]] * 5,
                YEAR: pd.Series([1990] * 10, dtype="Int16"),
            }
        )
    )
    users = UserFeatures.from_users(
        pd.DataFrame(
            {USER_IDX: range(40), GENDER: ["F"] * 40, AGE: [25] * 40, OCCUPATION: [1] * 40}
        )
    )
    cfg = TwoTowerConfig(
        id_dim=16,
        side_dim=4,
        hidden_dim=32,
        out_dim=16,
        max_history=5,
        temperature=0.1,
        batch_size=16,
        epochs=40,
        lr=1e-2,
        seed=0,
    )
    model = TwoTowerRecommender(cfg, items, users).fit(hist, 40, 10)
    scores = model.score(np.arange(40))
    seen = np.zeros((40, 10), dtype=bool)
    seen[hist[USER_IDX], hist[ITEM_IDX]] = True
    scores[seen] = -np.inf
    hit = np.mean([scores[u].argmax() in groups[u] for u in range(40)])
    assert hit >= 0.95
    vecs = model.item_vectors_
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)  # L2-normalized


# ------------------------------------------------------------------ FAISS


def _unit(n: int, d: int, seed: int) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_flat_search_matches_numpy_with_seen_filter() -> None:
    items, queries = _unit(300, 16, 0), _unit(20, 16, 1)
    seen = [np.arange(q % 7 * 3) for q in range(20)]
    got = search_excluding_seen(build_index("flat", items), queries, seen, k=10)
    scores = queries @ items.T
    for q in range(20):
        scores[q, seen[q]] = -np.inf
    expected = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    np.testing.assert_array_equal(got, expected)


def test_ivf_with_all_cells_probed_is_exact_and_hnsw_is_close() -> None:
    items, queries = _unit(2000, 16, 2), _unit(50, 16, 3)
    seen = [np.empty(0, dtype=np.int64)] * 50
    exact = search_excluding_seen(build_index("flat", items), queries, seen, k=20)
    ivf = build_index("ivf", items, nlist=16, nprobe=16)  # every cell scanned -> exhaustive
    assert overlap_at_k(search_excluding_seen(ivf, queries, seen, k=20), exact) == 1.0
    hnsw = build_index("hnsw", items, M=16, ef_construction=200, ef_search=200)
    assert overlap_at_k(search_excluding_seen(hnsw, queries, seen, k=20), exact) > 0.95


# ------------------------------------------------------------------ tuning knobs


def _tiny_fit(**overrides: object) -> TwoTowerRecommender:
    rng = np.random.default_rng(1)
    rows = [(u, int(i), True, t) for u in range(12) for t, i in enumerate(rng.permutation(20)[:6])]
    items = ItemFeatures.from_items(
        pd.DataFrame(
            {ITEM_IDX: range(20), GENRES: [["A"]] * 20, YEAR: pd.Series([1990] * 20, dtype="Int16")}
        )
    )
    users = UserFeatures.from_users(pd.DataFrame({USER_IDX: range(12)}))
    cfg = TwoTowerConfig(
        id_dim=8, side_dim=4, hidden_dim=16, out_dim=8, max_history=4, batch_size=8, epochs=2
    )
    model = TwoTowerRecommender(replace(cfg, **overrides), items, users)  # type: ignore[arg-type]
    return model.fit(_interactions(rows), 12, 20)


def test_disabling_user_id_makes_user_vectors_ignore_the_id_table() -> None:
    model = _tiny_fit(use_user_id=False, dropout=0.2, weight_decay=0.1)
    before = model.user_embeddings(np.arange(12))
    with torch.no_grad():
        model.net_.user_id.weight.normal_()  # scramble the (unused) ID embeddings
    np.testing.assert_allclose(model.user_embeddings(np.arange(12)), before, atol=1e-6)

    with_id = _tiny_fit(use_user_id=True)
    before = with_id.user_embeddings(np.arange(12))
    with torch.no_grad():
        with_id.net_.user_id.weight.normal_()
    assert not np.allclose(with_id.user_embeddings(np.arange(12)), before)


def test_numpy_user_tower_matches_torch() -> None:
    """Serving's numpy tower is a second implementation of the user tower, so it is pinned
    to the torch one: trained weights, dropout configured (identity in eval), ID on and off,
    partly padded and all-padding histories."""
    model = _tiny_fit(dropout=0.2)
    rng = np.random.default_rng(0)
    users = np.arange(12)
    tokens = rng.integers(1, 21, size=(12, 4))
    tokens[:4, :2] = PAD_TOKEN  # short histories (left-padded)
    tokens[4] = PAD_TOKEN  # no liked items at all
    use_id = users % 3 != 0
    torch_vecs = model.user_embeddings_from_history(users, tokens, use_id)
    numpy_vecs = model.user_embeddings_numpy(users, tokens, use_id)
    assert numpy_vecs.dtype == np.float32
    np.testing.assert_allclose(numpy_vecs, torch_vecs, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(numpy_vecs, axis=1), 1.0, atol=1e-6)


def test_numpy_user_tower_follows_a_refit() -> None:
    model = _tiny_fit()
    users, tokens = np.arange(12), np.ones((12, 4), dtype=np.int64)
    use_id = np.ones(12, dtype=bool)
    first = model.user_embeddings_numpy(users, tokens, use_id)
    model.fit(_interactions([(u, (u + 3) % 20, True, 0) for u in range(12)]), 12, 20)
    np.testing.assert_allclose(  # the export is redone, not served from the old weights
        model.user_embeddings_numpy(users, tokens, use_id),
        model.user_embeddings_from_history(users, tokens, use_id),
        atol=1e-6,
    )
    assert not np.allclose(model.user_embeddings_numpy(users, tokens, use_id), first)


def test_logq_alpha_zero_equals_no_correction() -> None:
    off = _tiny_fit(logq_correction=False)
    zero = _tiny_fit(logq_correction=True, logq_alpha=0.0)
    np.testing.assert_allclose(off.item_vectors_, zero.item_vectors_, atol=1e-6)
