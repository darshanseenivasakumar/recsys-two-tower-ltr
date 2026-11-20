"""Tune, evaluate and compare the Phase 2 baselines, logging everything to MLflow.

    python -m recsys.pipelines.baselines --data-config configs/data/ml-1m.yaml \
        --config configs/baselines.yaml [--models als bpr]

For each model:
  1. every hyperparameter combination -> fit on train, score on val (one nested MLflow run each)
  2. pick the best by the configured segment/metric (warm-user NDCG@10 by default)
  3. refit that config on train+val and score on test ONCE (the parent run)
Test numbers are never used to choose anything; that would turn test into a second
validation set and make the reported results optimistic.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import yaml

from recsys.config import DataConfig, load_data_config
from recsys.data.preprocess import load_processed
from recsys.data.split import SplitResult, load_split
from recsys.eval.protocol import Stage, eval_data_for_stage, recommend_with_scores, summarize
from recsys.models import build_model

logger = logging.getLogger(__name__)

# Columns of the comparison table: (label, stage, segment, metric).
TABLE_COLUMNS = [
    ("val warm NDCG@10", "val", "warm", "ndcg@10"),
    ("Recall@10", "test", "all", "recall@10"),
    ("NDCG@10", "test", "all", "ndcg@10"),
    ("HR@10", "test", "all", "hit_rate@10"),
    ("MRR@10", "test", "all", "mrr@10"),
    ("Recall@100", "test", "all", "recall@100"),
    ("Recall@200", "test", "all", "recall@200"),
    ("Coverage@10", "test", "all", "coverage@10"),
    ("ARP@10", "test", "all", "arp@10"),
    ("LongTail@10", "test", "all", "long_tail@10"),
]


@dataclass(frozen=True)
class EvalResult:
    summary: dict[str, dict[str, float]]
    fit_seconds: float
    score_seconds: float
    loss_history: list[float]


def expand_grid(fixed: dict[str, Any], grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of `grid` values, each merged over `fixed`."""
    if not grid:
        return [dict(fixed)]
    keys = list(grid)
    return [
        {**fixed, **dict(zip(keys, values, strict=True))}
        for values in itertools.product(*grid.values())
    ]


def mlflow_key(name: str) -> str:
    """MLflow metric names can't contain '@'."""
    return name.replace("@", "_at_")


def flatten(summary: dict[str, dict[str, float]], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_{segment}_{mlflow_key(metric)}": value
        for segment, stats in summary.items()
        for metric, value in stats.items()
    }


def evaluate(
    model_name: str,
    params: dict[str, Any],
    split: SplitResult,
    stage: Stage,
    n_users: int,
    n_items: int,
    k_max: int,
    ks: list[int],
) -> EvalResult:
    data = eval_data_for_stage(split, stage, n_items)
    model = build_model(model_name, params)

    start = time.perf_counter()
    model.fit(data.history, n_users, n_items)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    recs = recommend_with_scores(model.score, data, k_max)
    score_seconds = time.perf_counter() - start

    inner = getattr(model, "model", model)  # unwrap ColdStartFallback
    return EvalResult(
        summary=summarize(recs, data, ks),
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
        loss_history=list(getattr(inner, "loss_history_", [])),
    )


def _log_result(result: EvalResult, prefix: str) -> None:
    mlflow.log_metrics(flatten(result.summary, prefix))
    mlflow.log_metrics(
        {
            f"{prefix}_fit_seconds": result.fit_seconds,
            f"{prefix}_score_seconds": result.score_seconds,
        }
    )
    for step, loss in enumerate(result.loss_history):
        mlflow.log_metric(f"{prefix}_train_loss", loss, step=step)


def run_model(
    model_name: str,
    spec: dict[str, Any],
    split: SplitResult,
    data_cfg: DataConfig,
    n_users: int,
    n_items: int,
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    k_max, ks = int(eval_cfg["k_max"]), list(eval_cfg["ks"])
    segment, metric = eval_cfg["selection"]["segment"], eval_cfg["selection"]["metric"]
    candidates = expand_grid(spec.get("fixed", {}), spec.get("grid", {}))

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags(
            {
                "model": model_name,
                "dataset": data_cfg.dataset,
                "split": data_cfg.split.strategy,
                "selection": f"val_{segment}_{metric}",
            }
        )
        best_params: dict[str, Any] = {}
        best_val: EvalResult | None = None
        for params in candidates:
            with mlflow.start_run(run_name=f"{model_name}/val", nested=True):
                mlflow.log_params(params)
                result = evaluate(model_name, params, split, "val", n_users, n_items, k_max, ks)
                _log_result(result, "val")
            value = result.summary[segment][metric]
            logger.info("%-12s %s  val %s %s=%.4f", model_name, params, segment, metric, value)
            if best_val is None or value > best_val.summary[segment][metric]:
                best_params, best_val = params, result

        assert best_val is not None
        logger.info(
            "%-12s best %s -> refit on train+val, evaluate on test", model_name, best_params
        )
        test = evaluate(model_name, best_params, split, "test", n_users, n_items, k_max, ks)
        mlflow.log_params(best_params)
        _log_result(best_val, "val")
        _log_result(test, "test")

    return {
        "model": model_name,
        "params": best_params,
        "val": best_val.summary,
        "test": test.summary,
        "fit_seconds": test.fit_seconds,
    }


def comparison_markdown(rows: list[dict[str, Any]]) -> str:
    header = ["Model", *[c[0] for c in TABLE_COLUMNS], "Fit (s)", "Best params"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [row["model"]]
        for _, stage, segment, metric in TABLE_COLUMNS:
            value = row[stage][segment][metric]
            cells.append(f"{value:.1f}" if metric.startswith("arp") else f"{value:.4f}")
        params = ", ".join(f"{k}={v}" for k, v in row["params"].items() if k not in ("seed",))
        cells += [f"{row['fit_seconds']:.1f}", params or "-"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m recsys.pipelines.baselines")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="*", help="Subset of models to run (default: all)")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_cfg = load_data_config(args.data_config)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tables = load_processed(data_cfg.processed_path)
    n_users, n_items = len(tables.users), len(tables.items)
    split = load_split(data_cfg.split_path)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(f"{cfg['mlflow']['experiment_prefix']}-{data_cfg.dataset}")

    names = args.models or list(cfg["models"])
    rows = [
        run_model(name, cfg["models"][name], split, data_cfg, n_users, n_items, cfg["eval"])
        for name in names
    ]

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / f"baselines_{data_cfg.dataset}.json"
    # A partial run (--models) replaces only those models' rows in the existing report.
    previous = json.loads(report_json.read_text(encoding="utf-8")) if report_json.exists() else []
    rows = merge_rows(previous, rows, order=list(cfg["models"]))
    table = comparison_markdown(rows)
    (report_dir / f"baselines_{data_cfg.dataset}.md").write_text(table + "\n", encoding="utf-8")
    report_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(table)


def merge_rows(
    previous: list[dict[str, Any]], new: list[dict[str, Any]], order: list[str]
) -> list[dict[str, Any]]:
    """New rows win over previous ones for the same model; output follows `order`."""
    by_model = {row["model"]: row for row in previous} | {row["model"]: row for row in new}
    ranked = [by_model[m] for m in order if m in by_model]
    return ranked + [row for m, row in by_model.items() if m not in order]


if __name__ == "__main__":
    main()
