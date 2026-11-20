# Serving: ml-1m

Stack: `docker compose up --build` builds the API image (FastAPI, 4 uvicorn workers), Redis 7.4, a
one-shot seed job and the MLflow UI. The bundle (`artifacts/serving/ml-1m`) is the train+val snapshot:
the same two-tower, BPR, item statistics and LambdaRank model whose test numbers are in
`reports/ranker_ml-1m.md`.

## Request path (GET /recommend)

1. Redis `LRANGE recsys:u:<idx>`: the user's raw interaction log (re-ratings: latest wins).
2. No positive interaction (unknown or brand-new user): most-popular minus seen. Done.
3. Two-tower user vector from the liked-item history (same `encode_history` as training; the
   trained user tower run as numpy, `NumpyUserTower`, pinned to the torch one by tests), exact
   top-200 over 3,883 items; BPR top-200; union (~270 candidates); keep those some retriever
   ranked in its top 80 (~110; the score cap, picked on held-out val users).
4. `user_state()` + `feature_columns()`: the functions that built the ranker's training data,
   stacked straight into LightGBM's float matrix (no DataFrame on the request path).
5. LambdaRank scores from the trees compiled to native code (lleaves, 1 thread; bit-identical
   to LightGBM, checked at startup); MMR (lambda = 0.8, picked on held-out val users)
   reorders the top 50 into the top k.

A single positive POST /interaction switches a cold user to the personalized path on the next request
(the two-tower user vector is computed from history, so it needs no retraining). BPR's user factors do
NOT update online: until the next bundle, a user who becomes warm after the snapshot gets BPR's
popularity fallback as candidates, and a missing (NaN) `bpr_score` feature, exactly as in training
(`retrieval_features` is shared). Before that fix the feature carried raw popularity counts
(mean 266 vs 2.7 for real BPR scores), which the drift monitor flagged.

`GET /monitoring?last=5000` reads the last 5,000 sampled feature entries (10 candidate rows
each) from the Redis stream. Decoding each entry with its own `pd.read_json` cost ~9 ms, so
once the load tests had filled the stream the endpoint took 58 s; `EventLog.features` now
json-decodes all entries and builds one float matrix: 45 s -> 0.79 s for 50,000 rows,
identical output on 500 real entries (NaN pattern included), endpoint 1.1 s end to end. At
the default 5% sampling, 5,000 entries is ~100k requests, so a real deployment would have hit
this within hours.

## Training-serving parity (tests/test_serving.py)

For real test users, features built by the API path vs the rows the ranker was evaluated on:

| check (200 users, 22.8k scored candidate rows) | result |
|---|---|
| candidate sets (after the score cap) | identical |
| 40 of 47 features | identical (rtol 1e-9, NaN where NaN) |
| tt_score, bpr_score, x_sim_recent_max/mean, x_sim_last | within 2.4e-6 absolute (float32 GEMM batch-shape rounding: offline multiplies all ~270 candidates at once, serving the ~110 kept ones); same NaN pattern |
| tt_rank, bpr_rank | 3 of 46k cells differ, each by 1 (near-tie swaps from that rounding) |
| final top-10 shown to the user | identical for 200 / 200 users |

Parity compares offline and online *at the same decision time*, so it cannot see a
point-in-time mismatch (training asked at the window cutoff, serving mid-session). That
one was caught by the drift monitor and fixed by training on session queries
(`recsys/ranking/queries.py`, `reports/monitoring_ml-1m.md`).

## Latency

Profile of one warm request before the latency work (in-process, 1 thread): Redis 0.4 ms,
retrieval + features 2.8 ms, LightGBM + MMR 4.8 ms (predict alone 4.1 ms: 292 trees x ~270
rows, linear in both). What was tried, each judged by a rule fixed before the run:

| lever | result |
|---|---|
| fewer trees (learning rate 0.05-0.3 grid; serve the fastest model within 1% of the best held-out NDCG) | rejected: 10-190-tree models reach at best 0.1902 vs 0.1922; the 292-tree model stays |
| score fewer candidates (smallest "top-c of some retriever" within 1% of scoring all) | **c = 80**: ~110 rows instead of ~270, held-out NDCG 0.1907 vs 0.1922 |
| no pandas DataFrame on the request path | same values (unit test), Python calls per request 2.6M -> 1.2M per 300 requests |
| LightGBM threads per worker (all cores / 1 / 2; 6 workers) | 1 thread: all-cores prediction oversubscribes 8 cores across 4 workers |
| compile the trees (lleaves; served only if bit-identical to LightGBM) | **accepted**: predict 2.44 -> 0.49 ms per request (113-row median block, 1 thread); see below |

In-process, one thread, 200 test users: p50 6.8 ms, p95 9.3 ms (was 9.2 / 13.1 before the cap
and matrix path). The first ranker (23 trees) measured p50 13.6 ms with the pandas path.

Locust against the Docker stack on one 8-core laptop (Docker Desktop, Locust on the same machine),
90% GET /recommend / 10% POST /interaction, 0.2-1.0 s think time, 60 s per level, 0 failures,
4 uvicorn workers, `RANKER_NUM_THREADS=1`, LightGBM backend (before the compiled trees):

| simulated users | /recommend req/s | p50 | p95 | p99 | /interaction p50 / p95 |
|---|---|---|---|---|---|
| 20 | 29 | 11 ms | **24 ms** | 64 ms | 4 / 7 ms |
| 50 | 70 | 13 ms | **45 ms** | 92 ms | 5 / 13 ms |
| 75 | 101 | 19 ms | **130 ms** | 220 ms | 6 / 26 ms |
| 100 | 125 | 26 ms | **210 ms** | 360 ms | 8 / 48 ms |

Same setup before the cap and matrix path: p95 29 / 59 / 210 / 420 ms. First ranker (23
trees): p95 21 / 32 / 58 / 140 ms. p95 < 50 ms holds up to ~70 recommendations/s; beyond that
latency grows (not from the 8 cores saturating, as first assumed: see the py-spy profile). Thread sweep before the cap
(/recommend p95 at 50 / 75 users): all cores per worker 68 / 170 ms; 1 thread 62 / 110 ms;
2 threads 52 / 150 ms; 6 workers x 1 thread 56 / 180 ms. Runs near saturation vary (the
75-user p95 at 1 thread was 110 ms in that sweep, 210 ms in a repeat).

### Compiled trees (lleaves)

`recsys/serving/compiled.py` compiles `ranker.txt` with lleaves 1.3 (LLVM via llvmlite 0.44;
later llvmlite removed the legacy pass manager lleaves uses). Exactness on the served model:

| variant | predict, ms per request | vs LightGBM on all 320,660 test rows + 20k NaN/unseen-category rows |
|---|---|---|
| LightGBM 4.7, 1 thread | 2.44 | - |
| lleaves, default cache blocking (34 trees) | 0.25 | max diff 6.2e-15: blocking reorders the float sum |
| **lleaves, blocking off (served)** | 0.49 | **bit-identical** |

The binary is specialized to the host CPU, so it is compiled where it runs: the container
compiles it once at start (167 s on this laptop: with blocking off all 292 trees are one function
for LLVM to optimize, vs 33 s with the default blocking; before the workers fork) into a
compose volume, so later starts load it in seconds. The cache is keyed by model bytes +
lleaves/llvmlite versions + CPU name and features. Each worker loads it and compares it against
LightGBM on 20k drift-reference rows plus a copy with 15% of cells set to NaN; any difference,
import or compile error falls back to LightGBM with an ERROR log. `/health` reports
`ranker_backend`. Tests: bit-identity on a small LambdaRank model with a categorical feature
and NaN splits, cache reuse and keying, a planted stale binary refused, and identical
recommendations (items and scores) on 25 real users under both backends.

In-process, 200 test users, 1 thread, same run: LightGBM p50 6.1 / p95 9.0 ms (rank step
2.74 ms), compiled p50 **3.9 / 5.5 ms** (rank step 0.70 ms, the rest is MMR); identical top-10
for 200/200.

Locust A/B, interleaved: for each of two rounds, LightGBM then compiled, with Redis reseeded
before every 60 s level (POSTs from one level otherwise lengthen histories for the next).
0 failures in all 12 runs. /recommend:

| simulated users | backend | round 1 req/s, p50 / p95 / p99 | round 2 req/s, p50 / p95 / p99 |
|---|---|---|---|
| 50 | LightGBM | 69, 13 / 49 / 87 ms | 70, 13 / 56 / 120 ms |
| 50 | compiled | 70, 11 / **45** / 85 ms | 70, 11 / **45** / 88 ms |
| 75 | LightGBM | 99, 24 / 180 / 410 ms | 101, 18 / 130 / 210 ms |
| 75 | compiled | 100, 15 / **120** / 170 ms | 104, 13 / **63** / 120 ms |
| 100 | LightGBM | 130, 21 / **180** / 290 ms | 129, 24 / **150** / 240 ms |
| 100 | compiled | 126, 27 / 250 / 380 ms | 129, 22 / 170 / 270 ms |

Below saturation the compiled backend wins every pair, and it is the only one that keeps p95
under 50 ms at ~70 req/s in both rounds. At 100 users it lost both rounds, by less than the
round-to-round spread (up to 80 ms). The profile below explains it: the API is not CPU-bound
at that load, requests sharing a worker's GIL make the tail, and the spread between identical
runs on this machine is larger than the backend difference.

### Profile under load (py-spy)

py-spy 0.4.3 inside the API container (`cap_add: SYS_PTRACE` via a profiling-only compose
override), all 4 workers, 100 Hz, `--nonblocking`, 45 s at 100 simulated users (~140 req/s),
Redis reseeded first. Flamegraphs: `reports/profiles/flame_{compiled,lightgbm}_u100.svg`;
aggregation: `reports/profiles/analyze_stacks.py`, `latency_layers.py` (from the API's JSON
logs, steady-state window).

**The API is not CPU-bound.** At ~140 req/s the container uses 1.3-1.5 cores (compiled) and
1.8-2.2 cores (LightGBM) of 8 (`docker stats`), Redis 7%; each worker 0.4-0.7 cores of the
1 core its GIL allows; Locust printed no CPU warning. "The 8 cores saturate", which the
sections above assumed, was wrong.

Share of on-CPU samples, compiled backend (LightGBM run in brackets where it differs):

| where | share |
|---|---|
| uvicorn / httptools / starlette (HTTP, middleware) | 21% |
| two-tower user vector (torch) | 14% |
| ranker predict | **5%** (13%) |
| feature_columns | 8% |
| anyio threadpool handoff | 8% |
| Redis read | 7% |
| retrieval top-k, both sources | 6% |
| monitoring log (`_log`) | 6% |
| candidate union, MMR, user_state, rest of recommend() | 20% |
| JSON logging, response serialization, other | 4% |

(`Runner.run` / `Condition.wait` self-time, ~25%, is idle waiting py-spy counts as active.)

**The tail is inside each request, not in a queue in front of it.** Latency by layer, same run:

| ms | compiled p50 / p95 | LightGBM p50 / p95 |
|---|---|---|
| Locust (client) | 16 / 78 | 21 / 140 |
| middleware (handler + threadpool wait) | 11 / 60 | 17 / 118 |
| service total | 8 / 47 | 12 / 83 |
| of which Redis read | 1.3 / 6.3 | 1.6 / 13.7 |
| of which retrieval + features | 4.6 / 37 | 6.5 / 58 |

One request alone takes 3.9 / 5.5 ms; under load the service's own p95 is 6-9x its p50
while the cores idle. The Redis read, one socket round trip to an idle Redis, has a p95 8x
its median: nothing in it gets slower under load except re-acquiring the GIL afterwards.
All in-flight requests of a worker (anyio runs up to 40 at once) share one GIL, so each
request's wall time absorbs its neighbours' Python time.

**What removing that sharing does.** Interleaved runs, compiled backend, Redis reseeded, 0
failures, /recommend p95, round 1 / round 2:

| config | 100 users (~135 req/s) | 150 users (~190 req/s) |
|---|---|---|
| A: 4 workers x up to 40 concurrent requests (default) | 200 / 68 ms | 590 / 150 ms |
| B: 4 workers x 1 (`THREADPOOL_TOKENS=1`) | 210 / 380 ms | 470 / 390 ms |
| C: 8 workers x 1 | **53 / 55 ms** | 120 / 340 ms |

Capping concurrency alone (B) doesn't help: requests stop sharing a GIL but queue behind
4 processes instead. More processes with one request each (C) was the only configuration
that repeated: p95 53 / 55 ms at ~140 req/s in both rounds. At 150 users it won one round
and lost the other. The bigger finding is the spread: the same configuration varies up to
4x between rounds (A at 100 users: 200 vs 68 ms) while the machine is mostly idle, so this
laptop (Docker Desktop's VM, Locust on the same host) cannot resolve differences much
smaller than that above ~100 req/s. Defaults stay as they were; `THREADPOOL_TOKENS` and
`WEB_CONCURRENCY` are there for a test on a dedicated Linux host with the load generator on
another machine. (This also explains the 100-user A/B rows above: under py-spy the compiled
backend beat LightGBM there, 78 vs 140 ms, the reverse of both A/B rounds.)

### User tower in numpy

The profile put the two-tower user vector at 14% of samples: a dozen torch ops on [1, 50]
tensors, each paying torch's dispatch cost. `NumpyUserTower` (recsys/models/two_tower.py)
copies the trained user-tower weights once and runs the same math (embedding lookups,
masked mean, Linear -> ReLU -> Linear, L2 normalize; dropout is the identity in eval). Torch
stays the training and offline path; tests pin the two together.

| check | result |
|---|---|
| all 6,040 users, real histories, vs torch | max vector diff 1.5e-7, max retrieval-score diff 4.2e-7 (float32 summation order) |
| same top-200 candidate set as torch | 6,039 / 6,040 users (one near-tie at rank 200) |
| offline/online parity, 200 test users | candidate sets identical, 40 features exact, float32 scores within 2.4e-6 as before, 1 of 45.7k rank cells off by 1 (was 3), top-10 identical 200/200 |
| user vector, one user | torch 0.31 ms -> numpy **0.044 ms** |
| recommend(), same process, alternating per user (1,000 pairs) | p50 4.64 -> **3.99 ms**, retrieve + features 2.92 -> 2.32 ms, top-10 identical 1000/1000 |
| py-spy at 100 users | user tower 13.6% -> ~2% of samples (torch 0.3%) |

Under Locust the same 100-user run gave p95 100 ms (78 ms in the earlier profiling run):
inside this machine's run-to-run spread, so no load-test claim. The profile also caught one
worker at 1.00 core (its GIL's limit) while the other three ran 0.27-0.47: uvicorn workers
share one listening socket and the kernel does not balance connections evenly, so one
worker can queue while others idle, one more reason the tail is not about total CPU.

The remaining torch use on the request path is BPR's one matmul (`BPRArrays.scores`, 5% of
samples), kept as torch on purpose so its float32 scores match `BPRMF.score` offline.

### Balancing load across workers

`uvicorn --workers 4` binds one listening socket in the parent and the workers race to
`accept()` from it. Linux doesn't hand those out evenly, and Locust's keep-alive connections
stay on whichever worker accepted them, so the imbalance lasts the whole run (py-spy had one
worker at 1.00 core, its GIL limit, the others at 0.27-0.47). `recsys/serving/launch.py`
starts each worker on its own `SO_REUSEPORT` socket instead: the kernel assigns every new
connection to a socket by hashing its address. The parent only supervises (forwards SIGTERM
for graceful shutdown, restarts a dead worker, gives up on a crash loop); `docker stop`
exits in 1 s with code 0. `SERVER_SOCKETS=shared|per-worker` selects the mode; per-worker is
the default. Tests (`tests/test_launch.py`, Linux/macOS, run inside the serving image): 400
fresh connections reach all 4 workers with none above 40%, clean SIGTERM exit, and a worker
that fails on import is restarted at most 3 times before the launcher exits 1.

Interleaved A/B, each run in a fresh container with Redis reseeded; per-worker request share
from the `pid` field now on every JSON log line (max/min = busiest worker's share over the
quietest's; 1.00 is perfect):

| run | shared socket: max/min, server p95, Locust p95 | per-worker sockets: max/min, server p95, Locust p95 |
|---|---|---|
| 100 users, round 1 (~140 req/s) | 2.03, 65 ms, 73 ms | **1.70, 50 ms, 62 ms** |
| 100 users, round 2 | 1.90, 65 ms, 74 ms | **1.23, 53 ms, 66 ms** |
| 150 users, round 1 (~200 req/s) | 1.27, 139 ms, 160 ms | **1.28, 123 ms, 140 ms** |
| 150 users, round 2 | 1.50, 169 ms, 190 ms | **1.24**, 366 ms, 500 ms |
| 150 users, round 3 | 1.29, 533 ms, 610 ms | **1.26, 239 ms, 490 ms** |

Per-worker sockets are better or equal on balance in all five pairs and faster in four. The
exception (150 users, round 2) stayed balanced while throughput fell to 170 req/s and every
worker slowed together, the signature of a machine-wide slowdown; round 3, run to check it,
was slow for both modes and per-worker still led. 0 failures in every run.

What it does not fix: the kernel balances *connections*, not requests. With ~25 keep-alive
connections per worker, hashing alone leaves a few-connection difference (the 1.70 in round
1 is within that). Per-request balancing needs a proxy in front of the workers (nginx /
Envoy least-connections), or a load balancer that already does it in production.

Methodology note: the first Locust run used `HttpUser` (python-requests) and showed /interaction at
45 ms p50 for a single RPUSH. That was Nagle's algorithm + delayed ACK on requests' split
header/body writes, a client artifact; `FastHttpUser` measures 3 ms.

Raw Locust CSVs: `reports/loadtest/u{20,50,75,100}_*.csv` (LightGBM ladder) and
`reports/loadtest/ab_{lightgbm,lleaves}_r{1,2}_u{50,75,100}_*.csv` (backend A/B).
