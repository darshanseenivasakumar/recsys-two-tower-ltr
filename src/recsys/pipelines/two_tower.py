"""Train, select, evaluate and export the two-tower retrieval model; benchmark FAISS.

    python -m recsys.pipelines.two_tower --data-config configs/data/ml-1m.yaml \
        --config configs/two_tower.yaml

1. For every grid point: train on train, evaluate on val after EVERY epoch (MLflow curve).
   Each config's score = its best epoch's val warm Recall@200.
2. For each logQ setting, refit its best config on train+val for its best epoch count and
   evaluate once on test (exact dot-product search). The two rows are the logQ ablation.
3. Export the selected model, item embeddings and an HNSW index for serving (Phase 5).
4. Benchmark FAISS Flat / IVF / HNSW: Recall@200 vs ground truth, overlap with exact search,
   single-query latency. Repeat on a synthetic 300k-item catalog where ANN actually matters.

`grid` is either one dict (Cartesian product) or a list of dicts ("studies", each expanded
and concatenated), so single-factor ablations don't explode into a full factorial.
`--val-only` stops after step 1: tuning rounds never touch the test split.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import faiss
import mlflow
import numpy as np
import torch
import yaml

from recsys.config import load_data_config
from recsys.data.preprocess import load_processed
from recsys.data.split import SplitResult, load_split
from recsys.eval.protocol import (
    EvalData,
    Stage,
    eval_data_for_stage,
    recommend_with_scores,
    summarize,
)
from recsys.features.encoders import ItemFeatures, UserFeatures
from recsys.models.two_tower import TwoTowerConfig, TwoTowerRecommender
from recsys.pipelines.baselines import expand_grid, flatten, mlflow_key
from recsys.retrieval.ann import (
    build_index,
    overlap_at_k,
    search_excluding_seen,
    set_search_params,
    single_query_latency,
)

logger = logging.getLogger(__name__)


class Context:
    """Everything loaded once and shared by all runs."""

    def __init__(self, data_config: str, cfg: dict[str, Any]) -> None:
        self.data_cfg = load_data_config(data_config)
        tables = load_processed(self.data_cfg.processed_path)
        self.n_users, self.n_items = len(tables.users), len(tables.items)
        self.items = ItemFeatures.from_items(tables.items)
        self.users = UserFeatures.from_users(tables.users)
        self.split: SplitResult = load_split(self.data_cfg.split_path)
        self.cfg = cfg
        self.k_max = int(cfg["eval"]["k_max"])
        self.ks = list(cfg["eval"]["ks"])
        self.segment = cfg["eval"]["selection"]["segment"]
        self.metric = cfg["eval"]["selection"]["metric"]

    def eval_data(self, stage: Stage) -> EvalData:
        return eval_data_for_stage(self.split, stage, self.n_items)

    def model(self, config: TwoTowerConfig) -> TwoTowerRecommender:
        return TwoTowerRecommender(config, self.items, self.users)


Grid = dict[str, list[Any]] | list[dict[str, list[Any]]]


def grid_points(grid: Grid) -> list[dict[str, Any]]:
    """Overrides for every run: a dict is one Cartesian grid, a list is a union of grids."""
    studies = grid if isinstance(grid, list) else [grid]
    points: list[dict[str, Any]] = []
    for study in studies:
        for point in expand_grid({}, study or {}):
            if point not in points:
                points.append(point)
    return points


def run_label(overrides: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in overrides.items()) or "base"


def train_with_val_curve(ctx: Context, config: TwoTowerConfig) -> dict[str, Any]:
    """Train on train; score val after every epoch; return the best epoch and its summary."""
    data = ctx.eval_data("val")
    summaries: list[dict[str, dict[str, float]]] = []

    def on_epoch_end(epoch: int, model: TwoTowerRecommender) -> dict[str, float]:
        summary = summarize(recommend_with_scores(model.score, data, ctx.k_max), data, ctx.ks)
        summaries.append(summary)
        return {
            f"val_{ctx.segment}_{ctx.metric}": summary[ctx.segment][ctx.metric],
            f"val_{ctx.segment}_ndcg@10": summary[ctx.segment]["ndcg@10"],
        }

    start = time.perf_counter()
    model = ctx.model(config).fit(data.history, ctx.n_users, ctx.n_items, on_epoch_end)
    fit_seconds = time.perf_counter() - start
    curve = [s[ctx.segment][ctx.metric] for s in summaries]
    best = int(np.argmax(curve))
    return {
        "config": config,
        "best_epoch": best + 1,
        "best_value": curve[best],
        "summary": summaries[best],
        "epoch_metrics": model.epoch_metrics_,
        "fit_seconds": fit_seconds,
    }


def evaluate_on_test(
    ctx: Context, config: TwoTowerConfig
) -> tuple[TwoTowerRecommender, dict[str, Any]]:
    data = ctx.eval_data("test")
    start = time.perf_counter()
    model = ctx.model(config).fit(data.history, ctx.n_users, ctx.n_items)
    fit_seconds = time.perf_counter() - start
    summary = summarize(recommend_with_scores(model.score, data, ctx.k_max), data, ctx.ks)
    return model, {"summary": summary, "fit_seconds": fit_seconds}


def benchmark_ann(ctx: Context, model: TwoTowerRecommender) -> list[dict[str, Any]]:
    """Real catalog: accuracy and latency of Flat vs IVF vs HNSW on test users."""
    ann = ctx.cfg["ann"]
    k = int(ann["k"])
    data = ctx.eval_data("test")
    items = model.item_vectors_
    queries = model.user_embeddings(data.user_idx)

    flat = build_index("flat", items)
    exact = search_excluding_seen(flat, queries, data.seen, k)
    settings: list[tuple[str, Any, dict[str, Any]]] = [("flat", flat, {})]
    ivf = build_index("ivf", items, nlist=ann["ivf"]["nlist"])
    settings += [
        ("ivf", ivf, {"nlist": ann["ivf"]["nlist"], "nprobe": p}) for p in ann["ivf"]["nprobe"]
    ]
    hnsw = build_index(
        "hnsw", items, M=ann["hnsw"]["M"], ef_construction=ann["hnsw"]["ef_construction"]
    )
    settings += [
        ("hnsw", hnsw, {"M": ann["hnsw"]["M"], "ef_search": e}) for e in ann["hnsw"]["ef_search"]
    ]

    rows = []
    for kind, index, params in settings:
        set_search_params(index, **params)
        recs = search_excluding_seen(index, queries, data.seen, k)
        summary = summarize(recs, data, ks=[10, k])
        latency = single_query_latency(index, queries, k)
        rows.append(
            {
                "index": kind,
                "params": params,
                f"recall@{k}": summary["all"][f"recall@{k}"],
                "ndcg@10": summary["all"]["ndcg@10"],
                f"overlap@{k}": overlap_at_k(recs, exact),
                "p50_us": latency.p50_us,
                "p95_us": latency.p95_us,
                "p99_us": latency.p99_us,
            }
        )
        logger.info(
            "ANN %-5s %s recall@%d=%.4f overlap=%.4f p50=%.0fus p95=%.0fus",
            kind,
            params,
            k,
            rows[-1][f"recall@{k}"],
            rows[-1][f"overlap@{k}"],
            latency.p50_us,
            latency.p95_us,
        )
    return rows


def benchmark_synthetic(ctx: Context, model: TwoTowerRecommender) -> list[dict[str, Any]]:
    """Synthetic large catalog (jittered real item vectors): ANN speedup vs recall loss."""
    syn = ctx.cfg["ann"]["synthetic"]
    k = int(ctx.cfg["ann"]["k"])
    rng = np.random.default_rng(0)
    base = model.item_vectors_
    picks = rng.integers(0, len(base), size=int(syn["n_items"]))
    items = base[picks] + float(syn["noise"]) * rng.standard_normal(
        (len(picks), base.shape[1])
    ).astype(np.float32)
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    queries = model.user_embeddings(ctx.eval_data("test").user_idx)[:500]

    def timed_build(kind: str, **params: Any) -> tuple[Any, float]:
        start = time.perf_counter()
        index = build_index(kind, items, **params)
        return index, time.perf_counter() - start

    flat, flat_build = timed_build("flat")
    _, exact = flat.search(np.ascontiguousarray(queries), k)
    builds = {"flat": (flat, flat_build)}
    builds["ivf"] = timed_build("ivf", nlist=syn["ivf"]["nlist"])
    builds["hnsw"] = timed_build(
        "hnsw", M=syn["hnsw"]["M"], ef_construction=syn["hnsw"]["ef_construction"]
    )

    settings = [("flat", {})]
    settings += [("ivf", {"nlist": syn["ivf"]["nlist"], "nprobe": p}) for p in syn["ivf"]["nprobe"]]
    settings += [
        ("hnsw", {"M": syn["hnsw"]["M"], "ef_search": e}) for e in syn["hnsw"]["ef_search"]
    ]
    rows = []
    for kind, params in settings:
        index, build_s = builds[kind]
        set_search_params(index, **params)
        _, approx = index.search(np.ascontiguousarray(queries), k)
        latency = single_query_latency(index, queries, k)
        rows.append(
            {
                "index": kind,
                "params": params,
                "n_items": len(items),
                "build_s": build_s,
                f"overlap@{k}": overlap_at_k(approx, exact),
                "p50_us": latency.p50_us,
                "p95_us": latency.p95_us,
                "p99_us": latency.p99_us,
            }
        )
        logger.info(
            "SYN %-5s %s overlap=%.4f p50=%.0fus p95=%.0fus build=%.1fs",
            kind,
            params,
            rows[-1][f"overlap@{k}"],
            latency.p50_us,
            latency.p95_us,
            build_s,
        )
    return rows


def export(ctx: Context, model: TwoTowerRecommender, out_dir: Path) -> None:
    """Artifacts the serving layer (Phase 5) loads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.net_.state_dict(), out_dir / "model.pt")
    (out_dir / "config.json").write_text(
        json.dumps(asdict(model.config), indent=2), encoding="utf-8"
    )
    np.save(out_dir / "item_embeddings.npy", model.item_vectors_)
    ann = ctx.cfg["ann"]["hnsw"]
    index = build_index(
        "hnsw",
        model.item_vectors_,
        M=ann["M"],
        ef_construction=ann["ef_construction"],
        ef_search=max(ann["ef_search"]),
    )
    faiss.write_index(index, str(out_dir / "items_hnsw.faiss"))
    logger.info("Exported model, embeddings and HNSW index to %s", out_dir)


def _fmt(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def val_table(ctx: Context, val_runs: list[dict[str, Any]]) -> str:
    """One row per grid point: overrides vs the fixed config, best epoch and its val metrics."""
    rows = [
        [
            run_label(r["overrides"]),
            str(r["best_epoch"]),
            f"{r['best_value']:.4f}",
            f"{r['summary']['warm']['ndcg@10']:.4f}",
            f"{r['summary']['all']['coverage@100']:.4f}",
            f"{r['summary']['all']['arp@100']:.0f}",
            f"{r['epoch_metrics'][-1]['val_' + ctx.segment + '_' + ctx.metric]:.4f}",
            f"{r['fit_seconds']:.0f}",
        ]
        for r in val_runs
    ]
    header = [
        "overrides",
        "best epoch",
        f"val warm {ctx.metric}",
        "val warm NDCG@10",
        "val Coverage@100",
        "val ARP@100",
        f"{ctx.metric} at last epoch",
        "fit (s)",
    ]
    return "## Validation (train -> val), best epoch per config\n\n" + _fmt(rows, header)


def write_report(
    ctx: Context,
    val_runs: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    ann_rows: list[dict[str, Any]],
    syn_rows: list[dict[str, Any]],
    path: Path,
) -> str:
    k = int(ctx.cfg["ann"]["k"])
    sections = [f"# Two-tower retrieval: {ctx.data_cfg.dataset}\n"]
    sections.append(val_table(ctx, val_runs))
    if not test_rows:  # tuning round: validation only
        text = "\n".join(sections) + "\n"
        path.write_text(text, encoding="utf-8")
        return text

    baselines_path = Path("reports") / f"baselines_{ctx.data_cfg.dataset}.json"
    baselines = (
        json.loads(baselines_path.read_text(encoding="utf-8")) if baselines_path.exists() else []
    )
    rows = [(b["model"], b["test"]["all"]) for b in baselines if b["model"] != "random"]
    rows += [
        (
            f"two_tower ({run_label(r['overrides'])})",
            r["summary"]["all"],
        )
        for r in test_rows
    ]
    sections.append("\n## Test (train+val -> test), exact search, all users\n")
    sections.append(
        _fmt(
            [
                [
                    name,
                    *(
                        f"{s[m]:.4f}"
                        for m in (
                            "recall@100",
                            "recall@200",
                            "ndcg@10",
                            "recall@10",
                            "coverage@100",
                            "long_tail@100",
                        )
                    ),
                    f"{s['arp@100']:.0f}",
                ]
                for name, s in rows
            ],
            [
                "Model",
                "Recall@100",
                "Recall@200",
                "NDCG@10",
                "Recall@10",
                "Coverage@100",
                "LongTail@100",
                "ARP@100",
            ],
        )
    )

    sections.append(f"\n## FAISS on the real catalog ({ctx.n_items} items, test users, k={k})\n")
    sections.append(
        _fmt(
            [
                [
                    r["index"],
                    json.dumps(r["params"]),
                    f"{r[f'recall@{k}']:.4f}",
                    f"{r[f'overlap@{k}']:.4f}",
                    f"{r['p50_us']:.0f}",
                    f"{r['p95_us']:.0f}",
                    f"{r['p99_us']:.0f}",
                ]
                for r in ann_rows
            ],
            [
                "index",
                "params",
                f"Recall@{k}",
                f"overlap@{k} vs exact",
                "p50 µs",
                "p95 µs",
                "p99 µs",
            ],
        )
    )

    sections.append(
        f"\n## FAISS on a synthetic {syn_rows[0]['n_items']:,}-item catalog (ANN vs exact only)\n"
    )
    sections.append(
        _fmt(
            [
                [
                    r["index"],
                    json.dumps(r["params"]),
                    f"{r[f'overlap@{k}']:.4f}",
                    f"{r['p50_us']:.0f}",
                    f"{r['p95_us']:.0f}",
                    f"{r['p99_us']:.0f}",
                    f"{r['build_s']:.1f}",
                ]
                for r in syn_rows
            ],
            ["index", "params", f"overlap@{k} vs exact", "p50 µs", "p95 µs", "p99 µs", "build (s)"],
        )
    )
    sections.append("\nLatency: single query, single thread, includes Python call overhead.")
    text = "\n".join(sections) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m recsys.pipelines.two_tower")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, help="Override max epochs (smoke tests)")
    parser.add_argument(
        "--val-only", action="store_true", help="Tuning round: validation grid only, no test"
    )
    parser.add_argument("--report-name", help="Report file stem (default two_tower[_val]_<data>)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.epochs:
        cfg["model"]["epochs"] = args.epochs
    ctx = Context(args.data_config, cfg)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(f"{cfg['mlflow']['experiment_prefix']}-{ctx.data_cfg.dataset}")
    base = TwoTowerConfig(**cfg["model"])

    with mlflow.start_run(run_name="two_tower"):
        # 1. grid on validation, full per-epoch curves
        val_runs = []
        for overrides in grid_points(cfg["grid"]):
            config = replace(base, **overrides)
            with mlflow.start_run(run_name=f"val {run_label(overrides)}", nested=True):
                mlflow.log_params(asdict(config))
                run = train_with_val_curve(ctx, config) | {"overrides": overrides}
                for step, metrics in enumerate(run["epoch_metrics"]):
                    mlflow.log_metrics({mlflow_key(k): v for k, v in metrics.items()}, step=step)
                mlflow.log_metrics(flatten(run["summary"], "val_best"))
                mlflow.log_metric("best_epoch", run["best_epoch"])
            logger.info(
                "VAL %s best epoch %d: %s=%.4f",
                overrides,
                run["best_epoch"],
                ctx.metric,
                run["best_value"],
            )
            val_runs.append(run)

        stem = args.report_name or (
            f"two_tower{'_val' if args.val_only else ''}_{ctx.data_cfg.dataset}"
        )
        report_path = Path("reports") / f"{stem}.md"
        val_json = [
            {**{k: v for k, v in r.items() if k != "config"}, "config": asdict(r["config"])}
            for r in val_runs
        ]
        if args.val_only:
            text = write_report(ctx, val_runs, [], [], [], report_path)
            report_path.with_suffix(".json").write_text(
                json.dumps({"val": val_json}, indent=2, default=str), encoding="utf-8"
            )
            mlflow.log_artifact(str(report_path))
            print(text)
            return

        # 2. per logQ setting: best config on val -> refit train+val -> test once
        test_rows, selected = [], None
        for logq in (True, False):
            candidates = [r for r in val_runs if r["config"].logq_correction == logq]
            if not candidates:
                continue
            best = max(candidates, key=lambda r: r["best_value"])
            config = replace(best["config"], epochs=best["best_epoch"])
            model, result = evaluate_on_test(ctx, config)
            row = {
                "logq": logq,
                "overrides": best["overrides"],
                "temperature": config.temperature,
                "epochs": config.epochs,
                "val": best["best_value"],
                **result,
            }
            mlflow.log_metrics(flatten(result["summary"], f"test_logq_{'on' if logq else 'off'}"))
            test_rows.append(row)
            if selected is None or best["best_value"] > selected[0]:
                selected = (best["best_value"], model, row)
        assert selected is not None
        _, model, row = selected
        mlflow.log_params({"selected_" + k: v for k, v in asdict(model.config).items()})

        # 3. export; 4. ANN benchmarks
        out_dir = Path(cfg["artifacts_dir"]) / ctx.data_cfg.dataset
        export(ctx, model, out_dir)
        ann_rows = benchmark_ann(ctx, model)
        syn_rows = benchmark_synthetic(ctx, model)

        text = write_report(ctx, val_runs, test_rows, ann_rows, syn_rows, report_path)
        (report_path.with_suffix(".json")).write_text(
            json.dumps(
                {
                    "val": val_json,
                    "test": test_rows,
                    "ann": ann_rows,
                    "synthetic": syn_rows,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(report_path))
        print(text)


if __name__ == "__main__":
    main()
