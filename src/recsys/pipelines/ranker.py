"""Phase 4: candidates -> features -> LambdaRank -> MMR, evaluated on the temporal protocol.

    python -m recsys.pipelines.ranker --data-config configs/data/ml-1m.yaml \
        --config configs/ranker.yaml [--rebuild]

1. Build three windows (see configs/ranker.yaml). For each: fit both retrievers on the
   window's history only; turn the window into queries (recsys.ranking.queries): one at
   the history cutoff per user, plus, for the training windows, a few "session" queries
   at random moments inside the label period, with the user's history up to that moment
   (what the API sees mid-session). Per query: union of the retrievers' top-k, features
   as of the query's decision time, labels = positives after it. Cached to disk.
2. Train LambdaRank on w1 + 75% of w2's users (all their queries); early-stop and select
   hyperparameters on all queries of the other 25% (warm NDCG@10). A comparison ranker
   trained on cutoff queries only goes through the same selection and is exported too.
3. Evaluate once on w3 (test, cutoff queries only = the protocol every model used)
   against each retriever alone and the candidate-recall ceiling.
4. MMR lambda sweep (relevance vs diversity), feature importances, export the model.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from recsys.config import load_data_config
from recsys.data.preprocess import load_processed
from recsys.data.schema import TIMESTAMP
from recsys.data.split import load_split
from recsys.eval import metrics
from recsys.eval.protocol import EvalData, summarize
from recsys.features.encoders import ItemFeatures, UserFeatures
from recsys.features.history import encode_history
from recsys.features.ranking import ItemTable, UserBlock, compute_features, user_state
from recsys.models import build_model
from recsys.models.two_tower import TwoTowerConfig, TwoTowerRecommender
from recsys.pipelines.baselines import expand_grid, flatten
from recsys.ranking.candidates import (
    Candidates,
    label_candidates,
    merge_sources,
    retrieval_features,
    top_scored,
    topk_with_scores,
    within_cap,
)
from recsys.ranking.mmr import mmr_rerank
from recsys.ranking.queries import Queries, build_queries
from recsys.ranking.ranker import LambdaRankRanker, RankingTable

logger = logging.getLogger(__name__)
WINDOWS = ("w1", "w2", "w3")


# ------------------------------------------------------------------ windows


@dataclass(frozen=True)
class Window:
    history: pd.DataFrame  # everything retrievers and item statistics are fit on
    target: pd.DataFrame  # the label period, strictly after the history


def make_windows(split: Any, cut_quantile: float) -> dict[str, Window]:
    """history -> label-period windows; every label lies strictly after its history."""
    train = split.train
    cut = train[TIMESTAMP].quantile(cut_quantile, interpolation="higher")
    windows = {
        "w1": Window(train[train[TIMESTAMP] < cut], train[train[TIMESTAMP] >= cut]),
        "w2": Window(train, split.val),
        "w3": Window(pd.concat([train, split.val], ignore_index=True), split.test),
    }
    for name, w in windows.items():
        if w.history[TIMESTAMP].max() >= w.target[TIMESTAMP].min():
            raise AssertionError(f"window {name}: history overlaps its label period")
    return windows


def window_queries(
    windows: dict[str, Window], n_items: int, sessions_per_user: int, seed: int
) -> dict[str, Queries]:
    """Training windows get session queries; the test window keeps the offline protocol."""
    rng = np.random.default_rng(seed)
    return {
        name: build_queries(
            w.history, w.target, n_items, 0 if name == "w3" else sessions_per_user, rng
        )
        for name, w in windows.items()
    }


def subset(data: EvalData, rows: npt.NDArray[np.int64]) -> EvalData:
    return EvalData(
        user_idx=data.user_idx[rows],
        relevant=[data.relevant[r] for r in rows],
        seen=[data.seen[r] for r in rows],
        is_warm=data.is_warm[rows],
        item_popularity=data.item_popularity,
        history=data.history,
    )


# ------------------------------------------------------------------ window tables


@dataclass
class WindowTable:
    rows: npt.NDArray[np.int64]
    items: npt.NDArray[np.int64]
    features: pd.DataFrame
    labels: npt.NDArray[np.int64]
    retriever_recs: dict[str, npt.NDArray[np.int64]]  # each source's own top-k
    item_vectors: npt.NDArray[np.float32]  # the window's two-tower item vectors
    queries: pd.DataFrame  # Queries.meta(): user, as_of, at_cutoff, personalized, ...

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        frame = self.features.copy()
        frame["_row"], frame["_item"], frame["_label"] = self.rows, self.items, self.labels
        frame.to_parquet(out / "pairs.parquet", index=False)
        np.savez_compressed(out / "retriever_recs.npz", **self.retriever_recs)
        np.save(out / "item_vectors.npy", self.item_vectors)
        self.queries.to_parquet(out / "queries.parquet", index=False)

    @classmethod
    def load(cls, out: Path) -> WindowTable:
        frame = pd.read_parquet(out / "pairs.parquet")
        rows, items = frame.pop("_row").to_numpy(), frame.pop("_item").to_numpy()
        labels = frame.pop("_label").to_numpy()
        with np.load(out / "retriever_recs.npz") as recs:
            retriever_recs = {k: recs[k] for k in recs.files}
        return cls(
            rows,
            items,
            frame,
            labels,
            retriever_recs,
            np.load(out / "item_vectors.npy"),
            pd.read_parquet(out / "queries.parquet"),
        )


class Context:
    def __init__(self, data_config: str, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.data_cfg = load_data_config(data_config)
        tables = load_processed(self.data_cfg.processed_path)
        self.items_df = tables.items
        self.n_users, self.n_items = len(tables.users), len(tables.items)
        self.item_features = ItemFeatures.from_items(tables.items)
        self.user_features = UserFeatures.from_users(tables.users)
        wcfg = cfg["windows"]
        self.windows = make_windows(
            load_split(self.data_cfg.split_path), float(wcfg["extra_cut_quantile"])
        )
        self.queries = window_queries(
            self.windows, self.n_items, int(wcfg["sessions_per_user"]), int(wcfg["seed"])
        )
        self.out_dir = Path(cfg["artifacts_dir"]) / self.data_cfg.dataset
        retr = cfg["retrievers"]
        tt_cfg = json.loads(Path(retr["tt"]["two_tower_config"]).read_text(encoding="utf-8"))
        self.tt_config = TwoTowerConfig(**tt_cfg)
        baselines = json.loads(Path(retr["bpr"]["baselines_report"]).read_text(encoding="utf-8"))
        self.bpr_params = next(b["params"] for b in baselines if b["model"] == "bpr")
        self.bpr_test = next(b["test"] for b in baselines if b["model"] == "bpr")


def query_vectors(
    tt: TwoTowerRecommender, queries: Queries, batch_size: int = 4096
) -> npt.NDArray[np.float32]:
    """Two-tower user vectors from each query's own history: the API's exact path
    (encode_history over liked items up to the decision, ID only for users seen in fit)."""
    max_len = tt.config.max_history
    tokens = np.stack([encode_history(h.liked(), max_len) for h in queries.history])
    use_id = tt.known_users_[queries.user] & tt.config.use_user_id
    return np.vstack(
        [
            tt.user_embeddings_from_history(
                queries.user[s : s + batch_size],
                tokens[s : s + batch_size],
                use_id[s : s + batch_size],
            )
            for s in range(0, queries.n, batch_size)
        ]
    )


def build_window(ctx: Context, window: Window, queries: Queries) -> WindowTable:
    """Fit retrievers on the window's history; generate, featurize and label candidates
    for every query, each as of its own decision time."""
    retr = ctx.cfg["retrievers"]
    start = time.perf_counter()
    tt = TwoTowerRecommender(ctx.tt_config, ctx.item_features, ctx.user_features)
    tt.fit(window.history, ctx.n_users, ctx.n_items)
    bpr = build_model("bpr", ctx.bpr_params).fit(window.history, ctx.n_users, ctx.n_items)
    logger.info("retrievers fit in %.0fs", time.perf_counter() - start)

    data = queries.data  # rows are queries: score functions receive query ids
    qvec = query_vectors(tt, queries)
    score_fns = {
        "tt": lambda q: qvec[q] @ tt.item_vectors_.T,
        "bpr": lambda q: bpr.score(queries.user[q]),  # frozen factors, like the bundle
    }
    sources = {
        name: topk_with_scores(fn, data, int(retr[name]["k"])) for name, fn in score_fns.items()
    }
    cands: Candidates = merge_sources(sources)
    has_vector = {"bpr": bpr.is_warm_[queries.user]}  # type: ignore[attr-defined]
    retrieval = retrieval_features(cands, score_fns, data, has_vector)

    table = ItemTable.from_history(window.history, ctx.items_df)
    states = [
        user_state(h.items, h.ratings, h.timestamps, h.positive, table, as_of=t)
        for h, t in zip(queries.history, queries.as_of, strict=True)
    ]
    block = UserBlock.stack(states, queries.user, ctx.user_features)
    features = compute_features(block, cands.rows, cands.items, table, tt.item_vectors_, retrieval)
    labels = label_candidates(cands, data)
    logger.info(
        "%d queries (%d users), %d candidates (%.0f / query), %.1f%% positive",
        queries.n,
        len(np.unique(queries.user)),
        len(cands),
        len(cands) / queries.n,
        100 * labels.mean(),
    )
    return WindowTable(
        cands.rows,
        cands.items,
        features,
        labels,
        {name: recs for name, (recs, _) in sources.items()},
        tt.item_vectors_,
        queries.meta(),
    )


def load_or_build(ctx: Context, rebuild: bool) -> dict[str, WindowTable]:
    tables = {}
    for name in WINDOWS:
        path = ctx.out_dir / "windows" / name
        if (path / "queries.parquet").exists() and not rebuild:
            tables[name] = WindowTable.load(path)
            logger.info("loaded cached window %s (%d pairs)", name, len(tables[name].rows))
        else:
            logger.info("building window %s", name)
            tables[name] = build_window(ctx, ctx.windows[name], ctx.queries[name])
            tables[name].save(path)
    return tables


# ------------------------------------------------------------------ ranking + evaluation


def ranking_table(
    window: WindowTable, mask: npt.NDArray[np.bool_], query_offset: int
) -> RankingTable:
    return RankingTable(
        window.features[mask].reset_index(drop=True),
        window.labels[mask],
        window.rows[mask] + query_offset,
    )


def candidate_recall(
    window: WindowTable, data: EvalData, rows: npt.NDArray[np.int64] | None = None
) -> dict[str, float]:
    """Share of each query's held-out positives that made it into the candidate set."""
    hits = np.bincount(window.rows, weights=window.labels, minlength=data.n_users)
    n_rel = np.array([len(r) for r in data.relevant])
    recall = hits / np.maximum(n_rel, 1)
    rows = np.arange(data.n_users) if rows is None else rows
    return {
        "all": float(recall[rows].mean()),
        "warm": float(recall[rows][data.is_warm[rows]].mean()),
        "candidates_per_query": float(np.isin(window.rows, rows).sum() / len(rows)),
    }


def evaluate_scores(
    window: WindowTable,
    data: EvalData,
    scores: npt.NDArray[np.float64],
    rows: npt.NDArray[np.int64],
    k: int,
    ks: list[int],
) -> dict[str, dict[str, float]]:
    recs, _ = top_scored(window.rows, window.items, scores, data.n_users, k)
    return summarize(recs[rows], subset(data, rows), ks)


def evaluate_retrievers(
    window: WindowTable, data: EvalData, rows: npt.NDArray[np.int64], ks: list[int]
) -> dict[str, dict[str, dict[str, float]]]:
    k = max(ks)
    return {
        name: summarize(recs[rows, :k], subset(data, rows), ks)
        for name, recs in window.retriever_recs.items()
    }


def mmr_sweep(
    window: WindowTable,
    data: EvalData,
    scores: npt.NDArray[np.float64],
    rows: npt.NDArray[np.int64],
    genres: npt.NDArray[np.float64],
    mmr_cfg: dict[str, Any],
) -> list[dict[str, float]]:
    pool, k = int(mmr_cfg["pool"]), int(mmr_cfg["k"])
    cand, rel = top_scored(window.rows, window.items, scores, data.n_users, pool)
    cand, rel = cand[rows], rel[rows]
    sub = subset(data, rows)
    out = []
    for lam in mmr_cfg["lambdas"]:
        recs = mmr_rerank(cand, rel, window.item_vectors, k, float(lam))
        summary = summarize(recs, sub, [k])
        out.append(
            {
                "lambda": float(lam),
                f"ndcg@{k}": summary["all"][f"ndcg@{k}"],
                f"recall@{k}": summary["all"][f"recall@{k}"],
                f"coverage@{k}": summary["all"][f"coverage@{k}"],
                f"arp@{k}": summary["all"][f"arp@{k}"],
                f"long_tail@{k}": summary["all"][f"long_tail@{k}"],
                f"ild_emb@{k}": float(
                    np.nanmean(metrics.intra_list_diversity(recs, window.item_vectors, k))
                ),
                f"ild_genre@{k}": float(np.nanmean(metrics.intra_list_diversity(recs, genres, k))),
            }
        )
        logger.info("MMR %s", out[-1])
    return out


def predict_ms(
    ranker: LambdaRankRanker,
    window: WindowTable,
    rows: npt.NDArray[np.int64],
    n_queries: int = 200,
    repeats: int = 3,
) -> float:
    """Median single-thread LightGBM time to score one query's candidates, as the API does
    (one request = one query; RANKER_NUM_THREADS=1 in the deployed stack)."""
    picked = rows[np.linspace(0, len(rows) - 1, min(n_queries, len(rows))).astype(int)]
    order = np.argsort(window.rows, kind="stable")
    bounds = np.searchsorted(window.rows[order], [picked, picked + 1])
    frame = window.features[ranker.feature_names_]
    blocks = [frame.iloc[order[a:b]] for a, b in zip(*bounds, strict=True)]
    times = []
    booster, n_iter = ranker.booster_, ranker.booster_.best_iteration
    for block in blocks:
        booster.predict(block, num_iteration=n_iter, num_threads=1)  # warm caches
        t = time.perf_counter()
        for _ in range(repeats):
            booster.predict(block, num_iteration=n_iter, num_threads=1)
        times.append((time.perf_counter() - t) / repeats * 1e3)
    return float(np.median(times))


def fit_and_select(
    train: RankingTable,
    valid: RankingTable,
    window: WindowTable,
    data: EvalData,
    select_rows: npt.NDArray[np.int64],
    cfg: dict[str, Any],
) -> tuple[LambdaRankRanker, dict[str, Any], list[dict[str, Any]]]:
    """Grid over LightGBM params; early stopping on `valid`. Selection rule (fixed before
    looking at results): among grid points whose warm NDCG@10 on `select_rows` is within
    `tolerance` (relative) of the best one, take the one with the lowest measured predict
    time per request. With tolerance 0 this is plain best-NDCG selection."""
    lgbm, ks = cfg["lgbm"], list(cfg["eval"]["ks"])
    sel = cfg["eval"]["selection"]
    seg, metric, tol = sel["segment"], sel["metric"], float(sel.get("tolerance", 0.0))
    key = f"holdout_{seg}_{metric}"
    grid, models = [], []
    for params in expand_grid(lgbm["fixed"], lgbm["grid"]):
        start = time.perf_counter()
        ranker = LambdaRankRanker(
            params, int(lgbm["num_boost_round"]), int(lgbm["early_stopping_rounds"])
        ).fit(train, valid)
        fit_s = time.perf_counter() - start
        summary = evaluate_scores(
            window, data, ranker.predict(window.features), select_rows, max(ks), ks
        )
        row = {
            "params": {k: params[k] for k in lgbm["grid"]},
            "best_iteration": ranker.booster_.best_iteration,
            key: summary[seg][metric],
            "predict_ms": predict_ms(ranker, window, select_rows),
            "fit_seconds": fit_s,
        }
        logger.info("GRID %s", row)
        grid.append(row)
        models.append(ranker)
    best = max(r[key] for r in grid)
    eligible = [i for i, r in enumerate(grid) if r[key] >= (1 - tol) * best]
    pick = min(eligible, key=lambda i: grid[i]["predict_ms"])
    for i, r in enumerate(grid):
        r["within_tolerance"], r["selected"] = i in eligible, i == pick
    return models[pick], grid[pick], grid


def restrict(window: WindowTable, keep: npt.NDArray[np.bool_]) -> WindowTable:
    """The window's pairs where `keep` (query numbering unchanged)."""
    return WindowTable(
        window.rows[keep],
        window.items[keep],
        window.features[keep].reset_index(drop=True),
        window.labels[keep],
        window.retriever_recs,
        window.item_vectors,
        window.queries,
    )


def score_cap_sweep(
    window: WindowTable,
    data: EvalData,
    scores: npt.NDArray[np.float64],
    rows: npt.NDArray[np.int64],
    cfg: dict[str, Any],
) -> tuple[int | None, list[dict[str, Any]]]:
    """How few candidates can the ranker score? Rule (fixed before the run): the smallest
    cap whose warm NDCG@10 on `rows` is within `tolerance` of scoring every candidate."""
    ks = list(cfg["eval"]["ks"])
    seg, metric = cfg["eval"]["selection"]["segment"], cfg["eval"]["selection"]["metric"]
    tol = float(cfg["score_cap"]["tolerance"])
    sweep = []
    for cap in [None, *sorted(cfg["score_cap"]["caps"], reverse=True)]:
        keep = within_cap(window.features, cap)
        summary = evaluate_scores(restrict(window, keep), data, scores[keep], rows, max(ks), ks)
        sweep.append(
            {
                "cap": cap,
                "rows_per_query": float(np.isin(window.rows[keep], rows).sum() / len(rows)),
                f"{seg}_{metric}": summary[seg][metric],
            }
        )
    full = sweep[0][f"{seg}_{metric}"]
    ok = [r["cap"] for r in sweep[1:] if r[f"{seg}_{metric}"] >= (1 - tol) * full]
    cap = min(ok) if ok else None
    for r in sweep:
        r["selected"] = r["cap"] == cap
    logger.info("score cap sweep %s -> %s", sweep, cap)
    return cap, sweep


def _headline(summary: dict[str, dict[str, float]]) -> dict[str, float]:
    a = summary["all"]
    return {"ndcg@10": a["ndcg@10"], "recall@10": a["recall@10"], "recall@100": a["recall@100"]}


# ------------------------------------------------------------------ report


def _fmt(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


TEST_METRICS = ("ndcg@10", "recall@10", "recall@20", "recall@50", "recall@100")


def _metric_rows(named: list[tuple[str, dict[str, dict[str, float]]]], seg: str) -> list[list[str]]:
    rows = []
    for name, summary in named:
        s = summary[seg]
        cov = summary["all"].get("coverage@10", float("nan"))
        rows.append([name, *(f"{s[m]:.4f}" for m in TEST_METRICS), f"{cov:.4f}"])
    return rows


def importance_chart(importance: pd.DataFrame, path: Path, top: int = 20) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless; imported here so the module stays import-light
    import matplotlib.pyplot as plt

    data = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(data) + 1))
    ax.barh(data["feature"], 100 * data["gain_share"], color="#3366cc", height=0.6)
    ax.set_xlabel("share of total split gain (%)")
    ax.set_title(f"LambdaRank feature importance (top {len(data)} by gain)", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(result: dict[str, Any], path: Path, k_mmr: int) -> str:
    s = [f"# Ranker (LightGBM LambdaRank): {result['dataset']}\n"]
    s.append("## Training data (windows)\n")
    s.append(
        _fmt(
            [
                [
                    w["name"],
                    str(w["n_users"]),
                    str(w["n_cutoff_queries"]),
                    str(w["n_session_queries"]),
                    f"{w['candidates_per_query']:.0f}",
                    f"{w['positive_rate']:.3f}",
                    f"{w['candidate_recall_warm']:.4f}",
                    w["role"],
                ]
                for w in result["windows"]
            ],
            [
                "window",
                "users",
                "cutoff queries",
                "session queries",
                "candidates/query",
                "positive rate",
                "candidate recall (warm)",
                "role",
            ],
        )
    )
    s.append(
        "\nA cutoff query asks at the end of the window's history (the offline protocol); a "
        "session query asks at a random moment inside the label period, with the user's "
        "history up to that moment and frozen models, which is what the API serves."
    )
    s.append(
        "\n## Hyperparameters (all queries of the held-out w2 users, cutoff + session; "
        "warm NDCG@10)\n"
    )
    s.append(
        "Rule: among grid points within 1% of the best held-out NDCG@10, serve the one with "
        "the lowest single-thread predict time per request (median over 200 held-out queries).\n"
    )
    s.append(
        _fmt(
            [
                [
                    json.dumps(g["params"]),
                    str(g["best_iteration"]),
                    f"{g['holdout_warm_ndcg@10']:.4f}",
                    f"{g['predict_ms']:.2f}",
                    "**selected**" if g["selected"] else ("yes" if g["within_tolerance"] else ""),
                    f"{g['fit_seconds']:.0f}",
                ]
                for g in result["grid"]
            ],
            [
                "params",
                "trees",
                "holdout warm NDCG@10",
                "predict ms / request",
                "within tolerance",
                "fit (s)",
            ],
        )
    )
    s.append("\n## Training on session queries vs cutoff queries only (same selection)\n")
    s.append(
        _fmt(
            [
                [
                    c["training_queries"],
                    json.dumps(c["params"]),
                    str(c["best_iteration"]),
                    f"{c['holdout_cutoff']['ndcg@10']:.4f}",
                    f"{c['holdout_session']['ndcg@10']:.4f}",
                    f"{c['test']['ndcg@10']:.4f}",
                    f"{c['test']['recall@10']:.4f}",
                    f"{c['test']['recall@100']:.4f}",
                    f"{c['test']['coverage@10']:.4f}",
                ]
                for c in result["training_queries_comparison"]
            ],
            [
                "training queries",
                "params",
                "trees",
                "held-out cutoff NDCG@10",
                "held-out session NDCG@10",
                "test NDCG@10",
                "test Recall@10",
                "test Recall@100",
                "test Coverage@10",
            ],
        )
    )
    s.append(
        "\nThe test protocol asks once, at the snapshot cutoff; serving asks mid-session. "
        "They reward different rankers. The served ranker is the one trained on both kinds "
        "of query; the test-period replay (reports/monitoring_*.md) compares the two online."
    )
    header = [
        "Model",
        *(m.replace("ndcg", "NDCG").replace("recall", "Recall") for m in TEST_METRICS),
        "Coverage@10",
    ]
    blocks = (
        ("Held-out w2 users, cutoff queries", "holdout"),
        ("Held-out w2 users, session queries", "holdout_session"),
        ("Test (w3)", "test"),
    )
    for title, key in blocks:
        block = result[key]
        named = [
            ("two-tower alone", block["retrievers"]["tt"]),
            ("BPR alone", block["retrievers"]["bpr"]),
            ("LambdaRank on union", block["ranker"]),
        ]
        for seg in ("all", "warm"):
            n = int(block["ranker"][seg]["n_users"])
            s.append(f"\n## {title}, {seg} (n={n})\n")
            s.append(_fmt(_metric_rows(named, seg), header))
        cr = block["candidate_recall"]
        s.append(
            f"\nCandidate recall (ceiling for any reordering): all {cr['all']:.4f}, "
            f"warm {cr['warm']:.4f}."
        )
    bpr2 = result["bpr_phase2_test"]
    s.append(
        f"\nPhase 2 BPR test row for cross-check: NDCG@10 {bpr2['ndcg@10']:.4f}, "
        f"Recall@100 {bpr2['recall@100']:.4f}."
    )

    s.append("\n## Feature importance (gain)\n")
    s.append("![feature importance](figures/ranker_importance.png)\n")
    s.append(
        _fmt(
            [
                [r["feature"], f"{100 * r['gain_share']:.1f}%", str(int(r["splits"]))]
                for r in result["importance"][:25]
            ],
            ["feature", "gain share", "splits"],
        )
    )

    s.append("\n## Scoring cap (held-out w2 queries, cutoff + session)\n")
    s.append(
        "Score only candidates some retriever ranked in its top `cap`; kept candidates keep "
        "every feature value. Rule: smallest cap within 1% of scoring all of them.\n"
    )
    s.append(
        _fmt(
            [
                [
                    "all" if r["cap"] is None else str(r["cap"]),
                    f"{r['rows_per_query']:.0f}",
                    f"{r['warm_ndcg@10']:.4f}",
                    "**selected**" if r["selected"] else "",
                ]
                for r in result["score_cap_sweep"]
            ],
            ["cap", "rows scored / query", "held-out warm NDCG@10", ""],
        )
    )
    capped = result["training_queries_comparison"][0]
    t, tc = capped["test"], capped["test_capped"]
    s.append(
        f"\nTest, served ranker: all candidates NDCG@10 {t['ndcg@10']:.4f} / Recall@10 "
        f"{t['recall@10']:.4f}; capped at {result['score_cap']}: {tc['ndcg@10']:.4f} / "
        f"{tc['recall@10']:.4f}, coverage@10 {tc['coverage@10']:.4f}."
    )

    for title, key in (("held-out w2 users", "mmr_holdout"), ("test", "mmr_test")):
        s.append(f"\n## MMR re-ranking of the top-{result['mmr_pool']} ({title}, all users)\n")
        s.append(
            _fmt(
                [
                    [
                        f"{r['lambda']:.1f}",
                        *(
                            f"{r[f'{m}@{k_mmr}']:.4f}"
                            for m in (
                                "ndcg",
                                "recall",
                                "coverage",
                                "long_tail",
                                "ild_emb",
                                "ild_genre",
                            )
                        ),
                        f"{r[f'arp@{k_mmr}']:.0f}",
                    ]
                    for r in result[key]
                ],
                [
                    "lambda",
                    f"NDCG@{k_mmr}",
                    f"Recall@{k_mmr}",
                    f"Coverage@{k_mmr}",
                    f"LongTail@{k_mmr}",
                    f"ILD (embedding)@{k_mmr}",
                    f"ILD (genre)@{k_mmr}",
                    f"ARP@{k_mmr}",
                ],
            )
        )
    s.append(
        "\nILD = mean pairwise (1 - cosine) in the top list. The embedding version uses the same "
        "two-tower vectors MMR optimizes against; the genre version is an independent check."
    )
    text = "\n".join(s) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m recsys.pipelines.ranker")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rebuild", action="store_true", help="Ignore cached window tables")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ctx = Context(args.data_config, cfg)
    ks = list(cfg["eval"]["ks"])
    tables = load_or_build(ctx, args.rebuild)
    w1, w2, w3 = (tables[n] for n in WINDOWS)
    q1, q2, q3 = (ctx.queries[n] for n in WINDOWS)
    d2, d3 = q2.data, q3.data

    # Hold out 25% of w2's USERS (every query of theirs, so no user is on both sides).
    # Early stopping and selection use their cutoff queries: the protocol test is scored on.
    rng = np.random.default_rng(int(cfg["holdout"]["seed"]))
    users2 = np.unique(q2.user)
    held_users = rng.choice(users2, size=int(cfg["holdout"]["frac"] * len(users2)), replace=False)
    held = np.isin(q2.user, held_users)  # per query
    holdout_rows = np.flatnonzero(held & q2.at_cutoff)
    session_rows = np.flatnonzero(held & ~q2.at_cutoff)
    held_rows = np.flatnonzero(held)  # early stopping + selection: both request types

    def training_set(cutoff_only: bool) -> RankingTable:
        keep1 = q1.at_cutoff if cutoff_only else np.ones(q1.n, dtype=bool)
        keep2 = ~held & (q2.at_cutoff if cutoff_only else True)
        m1, m2 = keep1[w1.rows], keep2[w2.rows]
        return RankingTable(
            pd.concat([w1.features[m1], w2.features[m2]], ignore_index=True),
            np.concatenate([w1.labels[m1], w2.labels[m2]]),
            np.concatenate([w1.rows[m1], w2.rows[m2] + q1.n]),
        )

    train = training_set(cutoff_only=False)
    valid = ranking_table(w2, held[w2.rows], q1.n)
    logger.info(
        "ranker train: %d queries, %d pairs; valid: %d queries",
        len(np.unique(train.query)),
        len(train.labels),
        len(held_rows),
    )

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(f"{cfg['mlflow']['experiment_prefix']}-{ctx.data_cfg.dataset}")
    with mlflow.start_run(run_name="lambdarank"):
        ranker, chosen, grid = fit_and_select(train, valid, w2, d2, held_rows, cfg)
        # Comparison: the same model trained on cutoff queries only (the pre-fix setup).
        ranker_cut, chosen_cut, _ = fit_and_select(
            training_set(cutoff_only=True), valid, w2, d2, held_rows, cfg
        )

        w2_scores = ranker.predict(w2.features)
        test_scores = ranker.predict(w3.features)
        cap, cap_sweep = score_cap_sweep(w2, d2, w2_scores, held_rows, cfg)
        cap_cut, _ = score_cap_sweep(w2, d2, ranker_cut.predict(w2.features), held_rows, cfg)
        all_test = np.arange(d3.n_users)
        roles = ("train", f"train + {cfg['holdout']['frac']:.0%} holdout users", "test")
        holdout_summary = evaluate_scores(w2, d2, w2_scores, holdout_rows, max(ks), ks)
        comparison = []
        for name, model, best_row, c in (
            ("cutoff + session (served)", ranker, chosen, cap),
            ("cutoff only", ranker_cut, chosen_cut, cap_cut),
        ):
            sc2, sc3 = model.predict(w2.features), model.predict(w3.features)
            keep3 = within_cap(w3.features, c)
            test = evaluate_scores(w3, d3, sc3, all_test, max(ks), ks)
            served = evaluate_scores(restrict(w3, keep3), d3, sc3[keep3], all_test, max(ks), ks)
            comparison.append(
                {
                    "training_queries": name,
                    "params": best_row["params"],
                    "best_iteration": model.booster_.best_iteration,
                    "holdout_cutoff": _headline(
                        evaluate_scores(w2, d2, sc2, holdout_rows, max(ks), ks)
                    ),
                    "holdout_session": _headline(
                        evaluate_scores(w2, d2, sc2, session_rows, max(ks), ks)
                    ),
                    "test": _headline(test) | {"coverage@10": test["all"]["coverage@10"]},
                    "score_cap": c,
                    "test_capped": _headline(served)
                    | {"coverage@10": served["all"]["coverage@10"]},
                }
            )
        result: dict[str, Any] = {
            "dataset": ctx.data_cfg.dataset,
            "training_queries_comparison": comparison,
            "sessions_per_user": int(cfg["windows"]["sessions_per_user"]),
            "windows": [
                {
                    "name": n,
                    "n_users": int(len(np.unique(ctx.queries[n].user))),
                    "n_cutoff_queries": int(ctx.queries[n].at_cutoff.sum()),
                    "n_session_queries": int((~ctx.queries[n].at_cutoff).sum()),
                    "candidates_per_query": len(tables[n].rows) / ctx.queries[n].n,
                    "positive_rate": float(tables[n].labels.mean()),
                    "candidate_recall_warm": candidate_recall(tables[n], ctx.queries[n].data)[
                        "warm"
                    ],
                    "role": role,
                }
                for n, role in zip(WINDOWS, roles, strict=True)
            ],
            "grid": grid,
            "holdout": {
                "ranker": holdout_summary,
                "retrievers": evaluate_retrievers(w2, d2, holdout_rows, ks),
                "candidate_recall": candidate_recall(w2, d2, holdout_rows),
            },
            "holdout_session": {
                "ranker": evaluate_scores(w2, d2, w2_scores, session_rows, max(ks), ks),
                "retrievers": evaluate_retrievers(w2, d2, session_rows, ks),
                "candidate_recall": candidate_recall(w2, d2, session_rows),
            },
            "test": {
                "ranker": evaluate_scores(w3, d3, test_scores, all_test, max(ks), ks),
                "retrievers": evaluate_retrievers(w3, d3, all_test, ks),
                "candidate_recall": candidate_recall(w3, d3),
            },
            "bpr_phase2_test": ctx.bpr_test["all"],
            "importance": ranker.feature_importance().to_dict("records"),
            "mmr_pool": int(cfg["mmr"]["pool"]),
            "score_cap": cap,
            "score_cap_sweep": cap_sweep,
        }

        # MMR re-ranks what is actually served: the capped candidate set
        genres = ItemTable.from_history(d3.history, ctx.items_df).genres
        k2, k3 = within_cap(w2.features, cap), within_cap(w3.features, cap)
        result["mmr_holdout"] = mmr_sweep(
            restrict(w2, k2), d2, w2_scores[k2], holdout_rows, genres, cfg["mmr"]
        )
        result["mmr_test"] = mmr_sweep(
            restrict(w3, k3), d3, test_scores[k3], all_test, genres, cfg["mmr"]
        )

        mlflow.log_params({f"best_{k}": v for k, v in ranker.params.items()})
        mlflow.log_metrics(flatten(result["test"]["ranker"], "test_ranker"))
        mlflow.log_metrics(flatten(holdout_summary, "holdout_ranker"))

        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        ranker.save(ctx.out_dir / "ranker.txt")
        ranker_cut.save(ctx.out_dir / "ranker_cutoff_only.txt")
        (ctx.out_dir / "ranker_features.json").write_text(
            json.dumps(
                {
                    "features": ranker.feature_names_,
                    "params": ranker.params,
                    "score_cap": {"ranker.txt": cap, "ranker_cutoff_only.txt": cap_cut},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        figures = Path("reports") / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        importance_chart(ranker.feature_importance(), figures / "ranker_importance.png")
        report = Path("reports") / f"ranker_{ctx.data_cfg.dataset}.md"
        text = write_report(result, report, int(cfg["mmr"]["k"]))
        report.with_suffix(".json").write_text(
            json.dumps(result, indent=2, default=float), encoding="utf-8"
        )
        mlflow.log_artifact(str(report))
        mlflow.log_artifact(str(figures / "ranker_importance.png"))
        print(text)


if __name__ == "__main__":
    main()
