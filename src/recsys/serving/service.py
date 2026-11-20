"""Request-time recommendation: Redis log -> two retrievers -> features -> LambdaRank -> MMR.

Every step calls the SAME function the offline pipeline used (candidate union, pair
scores, user state, feature frame, implicit label). `tests/test_serving_parity.py` checks
that, for real test users, the features built here equal the ones the ranker was
evaluated on.

Time: features are computed "as of" `clock()`. The default clock is frozen at the bundle's
snapshot cutoff, which is what the offline evaluation used (MovieLens ends in 2003, so a
wall clock would put every time feature far outside the training range). A live system
passes a wall clock; the replay simulation passes the simulated event time.

Monitoring: with an EventLog attached, every request is logged (who, strategy, items,
latency) and a sampled share of requests also logs a few random candidate feature rows,
which the drift monitor compares against the ranker's training distribution.

Cold start: users we can't personalize for (unknown id, or no positive interaction yet)
get the most-popular list minus what they've already rated, the same rule as
ColdStartFallback offline. One positive rating is enough to switch to the full path,
because the two-tower user vector is built from history and needs no retraining.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.data.preprocess import positive_mask
from recsys.eval.metrics import PAD
from recsys.eval.protocol import EvalData
from recsys.features.history import encode_history
from recsys.features.ranking import UserBlock, feature_columns, feature_matrix, user_state
from recsys.ranking.candidates import (
    Candidates,
    merge_sources,
    retrieval_features,
    top_scored,
    topk_with_scores,
    within_cap,
)
from recsys.ranking.mmr import mmr_rerank
from recsys.serving.bundle import ServingBundle
from recsys.serving.store import EventLog, Interactions, InteractionStore

MAX_K = 50
RANKER_BACKENDS = ("lightgbm", "lleaves")
logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    user_id: str
    strategy: str  # "personalized" | "popular"
    items: list[dict[str, Any]]
    n_candidates: int
    timings_ms: dict[str, float] = field(default_factory=dict)
    item_idx: list[int] = field(default_factory=list)  # for monitoring, not the API


@dataclass
class PairFeatures:
    """Everything the ranker sees for one request (exposed for the parity test)."""

    candidates: Candidates
    matrix: npt.NDArray[np.float64]  # [n_candidates, n_features], the ranker's column order
    names: list[str]

    @property
    def features(self) -> pd.DataFrame:
        """As a DataFrame (built on demand: tests, sampled monitoring rows)."""
        return pd.DataFrame(self.matrix, columns=self.names)


class UnknownItemError(KeyError):
    pass


class UnknownUserError(KeyError):
    pass


class RecommenderService:
    def __init__(
        self,
        bundle: ServingBundle,
        store: InteractionStore,
        clock: Callable[[], float] | None = None,
        event_log: EventLog | None = None,
        feature_sample_rate: float = 0.0,
        feature_sample_k: int = 10,
        seed: int | None = None,
        ranker_threads: int = 0,
        ranker_backend: str = "lightgbm",
        ranker_cache_dir: Path | None = None,
    ) -> None:
        self.bundle = bundle
        self.store = store
        cutoff = float(bundle.manifest["cutoff_unix"])
        self.clock: Callable[[], float] = clock or (lambda: cutoff)
        self.event_log = event_log
        self.feature_sample_rate = feature_sample_rate
        self.feature_sample_k = feature_sample_k
        self._rng = np.random.default_rng(seed)
        # LightGBM predicts with every core by default: fastest for one request alone, but
        # N uvicorn workers x all cores oversubscribe the CPU under load. 0 = LightGBM default.
        self._predict_kw = {"num_threads": ranker_threads} if ranker_threads > 0 else {}
        self.booster = lgb.Booster(model_file=str(bundle.ranker_path))
        self.feature_names: list[str] = bundle.manifest["ranker_features"]
        self.ranker_backend, self._predict = self._ranker(ranker_backend, ranker_cache_dir)
        self.threshold = float(bundle.manifest["positive_threshold"])
        self.k = bundle.manifest["k"]
        self.mmr_pool = int(bundle.manifest["mmr"]["pool"])
        self.score_cap: int | None = bundle.manifest.get("score_cap")
        self.mmr_lambda = float(bundle.manifest["mmr"]["lambda"])
        catalog = bundle.catalog.sort_values("item_idx")
        self._item_ids = catalog["item_id"].astype(str).to_numpy()
        self._titles = catalog["title"].to_numpy()
        self._genres = catalog["genres"].to_numpy()

    def _ranker(
        self, backend: str, cache_dir: Path | None
    ) -> tuple[str, Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]:
        """(backend actually in use, predict function). "lleaves" serves the compiled trees
        only after they reproduce LightGBM bit for bit on the bundle's drift-reference
        rows; otherwise LightGBM, with the reason logged."""

        def lightgbm(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.asarray(self.booster.predict(x, **self._predict_kw), dtype=np.float64)

        if backend not in RANKER_BACKENDS:
            raise ValueError(f"ranker_backend must be one of {RANKER_BACKENDS}")
        if backend == "lightgbm":
            return "lightgbm", lightgbm
        from recsys.monitoring.reference import REFERENCE_FILE
        from recsys.serving.compiled import check_rows, verified_compiled

        ref = self.bundle.ranker_path.parent / REFERENCE_FILE
        if not ref.exists():
            logger.error("no reference rows to verify the compiled ranker, using LightGBM")
            return "lightgbm", lightgbm
        rows = pd.read_parquet(ref, columns=self.feature_names).to_numpy(np.float64)[:20_000]
        compiled = verified_compiled(
            self.bundle.ranker_path,
            cache_dir or self.bundle.ranker_path.parent / ".ranker-cache",
            check_rows(rows),
            self.booster,
        )
        return ("lleaves", compiled) if compiled is not None else ("lightgbm", lightgbm)

    # ------------------------------------------------------------------ feature path

    def _single_user_data(self, user_idx: int, seen: npt.NDArray[np.int64]) -> EvalData:
        return EvalData(
            user_idx=np.array([user_idx], dtype=np.int64),
            relevant=[np.empty(0, dtype=np.int64)],
            seen=[seen],
            is_warm=np.array([True]),
            item_popularity=np.zeros(self.bundle.n_items, dtype=np.int64),
            history=pd.DataFrame(),
        )

    def pair_features(
        self, user_idx: int, events: Interactions, as_of: float | None = None
    ) -> PairFeatures:
        b = self.bundle
        now = self.clock() if as_of is None else as_of
        positive = positive_mask(events.ratings, self.threshold)
        seen = np.unique(events.items)
        data = self._single_user_data(user_idx, seen)

        # two-tower: user vector from the liked-item history, same encoder as training
        order = np.lexsort((events.items, events.timestamps))
        liked = events.items[order][positive[order]]
        tokens = encode_history(liked, b.two_tower.config.max_history)[None, :]
        use_id = (
            np.array([bool(b.two_tower.known_users_[user_idx])]) & b.two_tower.config.use_user_id
        )
        # numpy copy of the trained user tower: same weights and math as the torch path the
        # offline windows used, without torch's per-op dispatch (7x faster for one user)
        user_vec = b.two_tower.user_embeddings_numpy(np.array([user_idx]), tokens, use_id)
        tt_dense = user_vec @ b.two_tower.item_vectors_.T  # exact search: 3.9k items
        bpr_dense = b.bpr.scores(user_idx)[None, :]
        score_fns: dict[str, Callable[[npt.NDArray[np.int64]], npt.NDArray[np.floating]]] = {
            "tt": lambda _users: tt_dense,
            "bpr": lambda _users: bpr_dense,
        }
        sources = {
            name: topk_with_scores(fn, data, int(self.k[name])) for name, fn in score_fns.items()
        }
        cands = merge_sources(sources)
        # BPR's factors freeze at bundle time: a user whose first like came later has no
        # vector, so BPR retrieves its popularity fallback but its score is missing (NaN),
        # exactly as in training (retrieval_features is shared).
        has_vector = {"bpr": np.array([b.bpr.has_vector(user_idx)])}
        retrieval = retrieval_features(cands, score_fns, data, has_vector)
        keep = within_cap(retrieval, self.score_cap)  # skip what no retriever ranked high
        cands = Candidates(
            cands.rows[keep], cands.items[keep], {k: v[keep] for k, v in retrieval.items()}
        )
        retrieval = cands.retrieval

        state = user_state(
            events.items, events.ratings, events.timestamps, positive, b.item_table, as_of=now
        )
        block = UserBlock.stack([state], [user_idx], b.user_features)
        columns = feature_columns(
            block,
            cands.rows,
            cands.items,
            b.item_table,
            b.two_tower.item_vectors_,
            retrieval,
        )
        return PairFeatures(cands, feature_matrix(columns, self.feature_names), self.feature_names)

    # ------------------------------------------------------------------ requests

    def popular_items(self, events: Interactions, k: int) -> npt.NDArray[np.int64]:
        """Cold-start list: most popular items the user hasn't interacted with."""
        seen = set(events.items.tolist())
        top = [i for i in self.bundle.popular[: k + len(seen)] if i not in seen][:k]
        return np.array(top, dtype=np.int64)

    def _log(
        self,
        now: float,
        user_idx: int | None,
        rec: Recommendation,
        pairs: PairFeatures | None,
    ) -> None:
        if self.event_log is None:
            return
        self.event_log.log_recommendation(
            now, user_idx, rec.strategy, rec.item_idx, rec.n_candidates, rec.timings_ms["total"]
        )
        if pairs is not None and self._rng.random() < self.feature_sample_rate:
            n = len(pairs.matrix)
            rows = np.sort(self._rng.choice(n, size=min(self.feature_sample_k, n), replace=False))
            self.event_log.log_features(now, pd.DataFrame(pairs.matrix[rows], columns=pairs.names))

    def _payload(self, items: npt.NDArray[np.int64], scores: list[float | None]) -> list[dict]:
        return [
            {
                "rank": r + 1,
                "item_id": self._item_ids[i],
                "title": self._titles[i],
                "genres": self._genres[i].split("|") if self._genres[i] else [],
                "score": s,
            }
            for r, (i, s) in enumerate(zip(items.tolist(), scores, strict=True))
        ]

    def recommend(self, user_id: str, k: int = 10, diversify: bool = True) -> Recommendation:
        if not 1 <= k <= MAX_K:
            raise ValueError(f"k must be in [1, {MAX_K}]")
        t0 = time.perf_counter()
        now = self.clock()
        timings: dict[str, float] = {}
        user_idx = self.bundle.user_index.get(str(user_id))
        events = (
            self.store.get(user_idx).latest_per_item()
            if user_idx is not None
            else Interactions.empty()
        )
        timings["store"] = (time.perf_counter() - t0) * 1e3

        if user_idx is None or not positive_mask(events.ratings, self.threshold).any():
            picks = self.popular_items(events, k)
            timings["total"] = (time.perf_counter() - t0) * 1e3
            rec = Recommendation(
                str(user_id),
                "popular",
                self._payload(picks, [None] * len(picks)),
                0,
                timings,
                picks.tolist(),
            )
            self._log(now, user_idx, rec, None)
            return rec

        t1 = time.perf_counter()
        pairs = self.pair_features(user_idx, events, as_of=now)
        timings["retrieve_and_features"] = (time.perf_counter() - t1) * 1e3

        t2 = time.perf_counter()
        scores = self._predict(pairs.matrix)
        rows, items = pairs.candidates.rows, pairs.candidates.items
        if diversify:
            pool, rel = top_scored(rows, items, scores, 1, self.mmr_pool)
            picks = mmr_rerank(pool, rel, self.bundle.two_tower.item_vectors_, k, self.mmr_lambda)[
                0
            ]
            score_of = dict(zip(items.tolist(), scores.tolist(), strict=True))
        else:
            picks, vals = top_scored(rows, items, scores, 1, k)
            picks = picks[0]
            score_of = dict(zip(picks.tolist(), vals[0].tolist(), strict=True))
        picks = picks[picks != PAD]
        timings["rank"] = (time.perf_counter() - t2) * 1e3
        timings["total"] = (time.perf_counter() - t0) * 1e3
        rec = Recommendation(
            str(user_id),
            "personalized",
            self._payload(picks, [round(score_of[i], 6) for i in picks.tolist()]),
            len(pairs.candidates),
            timings,
            picks.tolist(),
        )
        self._log(now, user_idx, rec, pairs)
        return rec

    def record(self, user_id: str, item_id: str, rating: float, timestamp: int | None) -> int:
        user_idx = self.bundle.user_index.get(str(user_id))
        if user_idx is None:
            raise UnknownUserError(user_id)
        item_idx = self.bundle.item_index.get(str(item_id))
        if item_idx is None:
            raise UnknownItemError(item_id)
        ts = int(self.clock()) if timestamp is None else int(timestamp)
        return self.store.append(user_idx, item_idx, float(rating), ts)
