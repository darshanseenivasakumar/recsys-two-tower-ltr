"""Aggregate py-spy collapsed stacks (--format raw --subprocesses) into per-step costs."""

import re
import sys
from collections import Counter, defaultdict

RATE, SECONDS = 100, 45

# inclusive buckets: a sample counts toward a step if any frame matches (first match wins
# within a group, so nested steps are attributed to the innermost listed one)
STEPS = [
    ("redis read (store.get)", r"InteractionStore\.get \("),
    ("two-tower user vector (torch)", r"user_embeddings_from_history|encode_history|torch/"),
    ("retrieval top-k (both sources)", r"topk_with_scores"),
    ("candidate union", r"merge_sources"),
    ("retrieval_features", r"retrieval_features"),
    ("user_state", r"user_state \("),
    ("feature_columns", r"feature_columns \("),
    ("feature_matrix", r"feature_matrix \("),
    ("ranker predict", r"RecommenderService\._ranker\.<locals>|lleaves|lightgbm"),
    ("MMR", r"mmr_rerank"),
    ("monitoring log (_log)", r"RecommenderService\._log \("),
    ("response payload", r"RecommenderService\._payload"),
    ("other in recommend()", r"RecommenderService\.recommend \("),
]
OUTSIDE = [
    ("JSON logging", r"logging/__init__\.py|recsys/serving/logging|json/encoder"),
    ("FastAPI response serialization", r"jsonable_encoder|serialize_response|JSONResponse|render \("),
    ("uvicorn/httptools/starlette", r"uvicorn/|httptools|starlette/|h11"),
    ("anyio threadpool handoff", r"anyio/"),
]


def parse(path):
    for line in open(path, encoding="utf-8"):
        stack, _, n = line.rstrip("\n").rpartition(" ")
        frames = stack.split(";")
        procs = [f for f in frames if f.startswith("process ")]
        pid = re.match(r"process (\d+)", procs[-1]).group(1)
        yield pid, frames[len(procs):], int(n)


def main(path):
    total, by_pid, by_thread = 0, Counter(), Counter()
    step, outside, leaf = Counter(), Counter(), Counter()
    for pid, frames, n in parse(path):
        total += n
        by_pid[pid] += n
        joined = ";".join(frames)
        in_request = "RecommenderService.recommend (" in joined
        by_thread["request threads (anyio WorkerThread)" if "WorkerThread.run" in joined
                  else "event loop / other"] += n
        if in_request:
            for name, pat in STEPS:
                hit = [i for i, f in enumerate(frames) if re.search(pat, f)]
                if hit:
                    step[name] += n
                    break
        else:
            for name, pat in OUTSIDE:
                if re.search(pat, joined):
                    outside[name] += n
                    break
            else:
                outside["other (outside recommend)"] += n
        leaf[re.sub(r":\d+\)$", ")", frames[-1]) if frames else "?"] += n

    wall = RATE * SECONDS
    print(f"samples {total} over {SECONDS}s @ {RATE} Hz -> {total / wall:.2f} cores busy (Python-visible)")
    for pid, n in sorted(by_pid.items()):
        print(f"  pid {pid}: {n / wall:.2f} cores")
    for k, n in by_thread.most_common():
        print(f"  {k}: {n / total:.0%}")
    req = sum(step.values())
    print(f"\ninside recommend(): {req / total:.0%} of samples")
    for k, n in step.most_common():
        print(f"  {k:32s} {n / total:6.1%}")
    print(f"\noutside recommend(): {1 - req / total:.0%}")
    for k, n in outside.most_common():
        print(f"  {k:32s} {n / total:6.1%}")
    print("\ntop self-time frames")
    for k, n in leaf.most_common(18):
        print(f"  {n / total:6.1%}  {k[:110]}")


main(sys.argv[1])
