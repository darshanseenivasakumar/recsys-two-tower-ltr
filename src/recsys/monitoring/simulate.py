"""Replay the test period through the live serving path to see drift and decay over time.

    python -m recsys.monitoring.simulate --bundle artifacts/serving/ml-1m \
        --data-config configs/data/ml-1m.yaml --redis-url redis://localhost:6379/1

The bundle is frozen at the end of val. We walk the real test-period interactions in time
order with a simulated clock: before a user's event, with probability `rec_prob`, they
request recommendations (logged exactly as the API logs them); then the event is written
to Redis exactly as POST /interaction would. Because the future is known, each request
can be scored against what the user actually liked in the next 7 days, for the served
list AND, counterfactually, for the most-popular list at the same moment.

This is the online picture offline evaluation can't give: how fast a frozen model and its
batch features go stale, and whether the drift monitor notices before accuracy drops.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys.config import load_data_config
from recsys.data.preprocess import positive_mask
from recsys.data.schema import ITEM_IDX, RATING, TIMESTAMP, USER_IDX
from recsys.data.split import load_split
from recsys.features.ranking import SECONDS_PER_DAY
from recsys.monitoring.reference import REFERENCE_FILE
from recsys.monitoring.report import summarize_traffic
from recsys.serving.bundle import ServingBundle
from recsys.serving.service import RecommenderService
from recsys.serving.store import EventLog, InteractionStore

logger = logging.getLogger(__name__)

# Buckets of snapshot age (days since the bundle's cutoff).
AGE_BUCKETS = [(0, 7), (7, 30), (30, 90), (90, 180), (180, 365), (365, 10_000)]
HORIZON_DAYS = 7


class SimClock:
    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def future_hits(
    requests: pd.DataFrame, positives: pd.DataFrame, horizon_days: float
) -> pd.DataFrame:
    """For each request (user, ts, served, popular): the user's likes in [ts, ts + horizon],
    and whether each list contains at least one of them."""
    by_user = {
        u: (g[TIMESTAMP].to_numpy(), g[ITEM_IDX].to_numpy())
        for u, g in positives.sort_values(TIMESTAMP).groupby(USER_IDX)
    }
    out = []
    for r in requests.itertuples(index=False):
        ts_arr, items = by_user.get(r.user, (np.empty(0), np.empty(0, np.int64)))
        lo, hi = np.searchsorted(ts_arr, [r.ts, r.ts + horizon_days * SECONDS_PER_DAY], "left")
        future = set(items[lo:hi].tolist())
        out.append(
            {
                "n_future_likes": len(future),
                "hit_served": bool(future & set(r.served)),
                "hit_popular": bool(future & set(r.popular)),
            }
        )
    return pd.concat([requests.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def run(
    bundle_dir: Path,
    data_config: str,
    redis_url: str,
    rec_prob: float,
    seed: int,
    max_events: int | None,
) -> dict[str, Any]:
    bundle = ServingBundle.load(bundle_dir)
    store = InteractionStore.from_url(redis_url)
    store.seed(pd.read_parquet(bundle_dir / "history.parquet"), bundle.manifest["version"])
    log = EventLog(store.client)
    log.clear()
    cutoff = float(bundle.manifest["cutoff_unix"])
    clock = SimClock(cutoff)
    svc = RecommenderService(
        bundle, store, clock=clock, event_log=log, feature_sample_rate=1.0, seed=seed
    )
    idx_to_id = {v: k for k, v in bundle.user_index.items()}
    threshold = float(bundle.manifest["positive_threshold"])

    test = load_split(load_data_config(data_config).split_path).test
    test = test.sort_values([TIMESTAMP, USER_IDX, ITEM_IDX], kind="stable")
    if max_events:
        test = test.head(max_events)
    rng = np.random.default_rng(seed)
    asks = rng.random(len(test)) < rec_prob
    requests = []
    start = time.perf_counter()
    for ask, user, item, rating, ts in zip(
        asks,
        test[USER_IDX].to_numpy(),
        test[ITEM_IDX].to_numpy(),
        test[RATING].to_numpy(),
        test[TIMESTAMP].to_numpy(),
        strict=True,
    ):
        clock.now = float(ts)
        if ask:
            rec = svc.recommend(idx_to_id[int(user)], k=10)
            events = store.get(int(user)).latest_per_item()
            requests.append(
                {
                    "user": int(user),
                    "ts": int(ts),
                    "strategy": rec.strategy,
                    "served": rec.item_idx,
                    "popular": svc.popular_items(events, 10).tolist(),
                }
            )
        store.append(int(user), int(item), float(rating), int(ts))
    logger.info(
        "replayed %d events, %d requests in %.0fs",
        len(test),
        len(requests),
        time.perf_counter() - start,
    )

    positives = test[positive_mask(test[RATING].to_numpy(), threshold)]
    scored = future_hits(pd.DataFrame(requests), positives, HORIZON_DAYS)
    scored["age_days"] = (scored["ts"] - cutoff) / SECONDS_PER_DAY

    recs_log = log.recommendations()
    feats_log = log.features()
    reference = pd.read_parquet(bundle_dir / REFERENCE_FILE)
    names = bundle.manifest["ranker_features"]
    popularity = np.expm1(bundle.item_table.log_pop)

    buckets = []
    for lo, hi in AGE_BUCKETS:
        sel = (scored["age_days"] >= lo) & (scored["age_days"] < hi)
        lo_ts, hi_ts = cutoff + lo * SECONDS_PER_DAY, cutoff + hi * SECONDS_PER_DAY
        window_recs = recs_log[(recs_log["ts"] >= lo_ts) & (recs_log["ts"] < hi_ts)]
        window_feats = (
            feats_log[(feats_log["_ts"] >= lo_ts) & (feats_log["_ts"] < hi_ts)]
            if not feats_log.empty
            else feats_log
        )
        if not sel.any():
            continue
        active = scored[sel & (scored["n_future_likes"] > 0)]
        summary = summarize_traffic(
            window_recs, window_feats, reference, names, popularity, cutoff, top=8
        )
        buckets.append(
            {
                "age_days": [lo, hi],
                "n_requests": int(sel.sum()),
                "n_with_future_likes": len(active),
                "hit_served": float(active["hit_served"].mean()) if len(active) else None,
                "hit_popular": float(active["hit_popular"].mean()) if len(active) else None,
                "summary": summary,
            }
        )
        logger.info(
            "bucket %s: %s", (lo, hi), {k: v for k, v in buckets[-1].items() if k != "summary"}
        )
    return {
        "bundle_version": bundle.manifest["version"],
        "n_events": len(test),
        "n_requests": len(requests),
        "rec_prob": rec_prob,
        "horizon_days": HORIZON_DAYS,
        "buckets": buckets,
        "overall": {
            "hit_served": float(scored.loc[scored.n_future_likes > 0, "hit_served"].mean()),
            "hit_popular": float(scored.loc[scored.n_future_likes > 0, "hit_popular"].mean()),
        },
    }


def _age_label(lo: int, hi: int) -> str:
    return f"{lo}+" if hi >= 10_000 else f"{lo}-{hi}"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def write_report(result: dict[str, Any], path: Path, chart: Path) -> str:
    lines = [
        "# Monitoring replay: ml-1m test period through the live serving path\n",
        f"Bundle {result['bundle_version']} (frozen at the end of val). Replayed "
        f"{result['n_events']:,} test-period interactions in time order; before each one the "
        f"user requested recommendations with probability {result['rec_prob']} "
        f"({result['n_requests']:,} requests). Hit = the 10-item list contains at least one "
        f"movie the user liked within the next {result['horizon_days']} days (requests followed "
        "by no like are excluded). 'Popular' is the most-popular list for the same user at the "
        "same moment, evaluated on the same future.\n",
        f"![hit rate by snapshot age]({chart.as_posix().split('reports/')[-1]})\n",
        "| snapshot age (days) | requests | with future likes | hit@10 served | "
        "hit@10 popular | coverage | avg popularity | max PSI (noise floor) | "
        "major / moderate features |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for b in result["buckets"]:
        s = b["summary"]
        exp = s.get("exposure_personalized", {})
        avg_pop = exp.get("avg_popularity", float("nan"))
        drift = s.get("drift")
        drift_txt = f"{drift['max_psi']:.2f} ({drift['noise_floor_psi']:.3f})" if drift else "-"
        counts = f"{drift['n_major']} / {drift['n_moderate']}" if drift else "-"
        lines.append(
            f"| {_age_label(*b['age_days'])} | {b['n_requests']} | "
            f"{b['n_with_future_likes']} | {_pct(b['hit_served'])} | {_pct(b['hit_popular'])} | "
            f"{_pct(exp.get('catalog_coverage'))} | {avg_pop:.0f} | "
            f"{drift_txt} | {counts} |"
        )
    lines.append(
        f"\nOverall hit@10 (7 days): served {_pct(result['overall']['hit_served'])}, "
        f"popular {_pct(result['overall']['hit_popular'])}."
    )
    for b in result["buckets"]:
        drift = b["summary"].get("drift")
        if not drift:
            continue
        lines.append(
            f"\n### Most drifted features, snapshot age {b['age_days'][0]}-"
            f"{b['age_days'][1]} days ({drift['n_rows']} sampled rows)\n"
        )
        lines.append("| feature | PSI | status | training mean | serving mean |")
        lines.append("|---|---|---|---|---|")
        for r in drift["top"]:
            lines.append(
                f"| {r['feature']} | {r['psi']:.3f} | {r['status']} | {r['ref_mean']:.3g} | "
                f"{r['cur_mean']:.3g} |"
            )
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def hit_rate_chart(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [b for b in result["buckets"] if b["n_with_future_likes"] >= 30]
    labels = [_age_label(*b["age_days"]) for b in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for key, name, color in (
        ("hit_served", "served (LambdaRank + MMR)", "#3366cc"),
        ("hit_popular", "most-popular", "#999999"),
    ):
        y = [100 * b[key] for b in rows]
        ax.plot(x, y, marker="o", markersize=6, linewidth=2, color=color, label=name)
        ax.annotate(
            name,
            (x[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color="#333333",
        )
    ax.set_xticks(x, labels)
    ax.set_xlabel("snapshot age (days since the model's training data ends)")
    ax.set_ylabel("hit@10 within 7 days (%)")
    ax.set_title("Online hit rate as the frozen model ages", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m recsys.monitoring.simulate")
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/serving/ml-1m"))
    parser.add_argument("--data-config", default="configs/data/ml-1m.yaml")
    parser.add_argument("--redis-url", default="redis://localhost:6379/1")
    parser.add_argument("--rec-prob", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--report-name", help="default: monitoring_<dataset>")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(
        args.bundle, args.data_config, args.redis_url, args.rec_prob, args.seed, args.max_events
    )
    dataset = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))["dataset"]
    name = args.report_name or f"monitoring_{dataset}"
    chart = Path("reports") / "figures" / f"{name.replace(f'_{dataset}', '')}_hit_rate.png"
    hit_rate_chart(result, chart)
    report = Path("reports") / f"{name}.md"
    print(write_report(result, report, chart))
    report.with_suffix(".json").write_text(
        json.dumps(result, indent=2, default=float), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
