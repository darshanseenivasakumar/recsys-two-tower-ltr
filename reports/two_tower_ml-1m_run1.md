# Two-tower retrieval: ml-1m

## Validation (train -> val), best epoch per config

| temperature | logQ | best epoch | val warm recall@200 | val warm NDCG@10 | val Coverage@10 | fit (s, all epochs) |
|---|---|---|---|---|---|---|
| 0.05 | True | 6 | 0.4288 | 0.0880 | 0.3309 | 201 |
| 0.05 | False | 19 | 0.1640 | 0.0235 | 0.6366 | 197 |
| 0.1 | True | 2 | 0.4276 | 0.1070 | 0.2495 | 200 |
| 0.1 | False | 20 | 0.1724 | 0.0260 | 0.6255 | 200 |

## Test (train+val -> test), exact search, all users

| Model | Recall@100 | Recall@200 | NDCG@10 | Recall@10 | Coverage@100 | LongTail@100 | ARP@100 |
|---|---|---|---|---|---|---|---|
| most_popular | 0.2700 | 0.4060 | 0.2190 | 0.0666 | 0.1455 | 0.0052 | 1322 |
| item_knn | 0.2896 | 0.4222 | 0.2486 | 0.0637 | 0.2073 | 0.0068 | 1264 |
| als | 0.3265 | 0.4669 | 0.2568 | 0.0713 | 0.2004 | 0.0059 | 1254 |
| bpr | 0.3396 | 0.4954 | 0.2624 | 0.0756 | 0.4494 | 0.0840 | 1094 |
| two_tower (logQ=on, T=0.05) | 0.2558 | 0.4055 | 0.1574 | 0.0481 | 0.7438 | 0.2648 | 814 |
| two_tower (logQ=off, T=0.1) | 0.0921 | 0.1608 | 0.0392 | 0.0126 | 1.0000 | 0.8078 | 226 |

## FAISS on the real catalog (3883 items, test users, k=200)

| index | params | Recall@200 | overlap@200 vs exact | p50 µs | p95 µs | p99 µs |
|---|---|---|---|---|---|---|
| flat | {} | 0.4055 | 1.0000 | 78 | 90 | 122 |
| ivf | {"nlist": 64, "nprobe": 1} | 0.0855 | 0.1526 | 10 | 12 | 13 |
| ivf | {"nlist": 64, "nprobe": 2} | 0.1547 | 0.2685 | 14 | 18 | 22 |
| ivf | {"nlist": 64, "nprobe": 4} | 0.2591 | 0.4482 | 19 | 24 | 32 |
| ivf | {"nlist": 64, "nprobe": 8} | 0.3632 | 0.6792 | 27 | 30 | 32 |
| ivf | {"nlist": 64, "nprobe": 16} | 0.3988 | 0.8854 | 34 | 38 | 45 |
| hnsw | {"M": 32, "ef_search": 16} | 0.2240 | 0.4027 | 30 | 39 | 44 |
| hnsw | {"M": 32, "ef_search": 32} | 0.3092 | 0.5792 | 46 | 54 | 63 |
| hnsw | {"M": 32, "ef_search": 64} | 0.3772 | 0.7830 | 67 | 76 | 85 |
| hnsw | {"M": 32, "ef_search": 128} | 0.4021 | 0.9351 | 98 | 117 | 143 |
| hnsw | {"M": 32, "ef_search": 256} | 0.4056 | 0.9927 | 163 | 178 | 201 |

## FAISS on a synthetic 300,000-item catalog (ANN vs exact only)

| index | params | overlap@200 vs exact | p50 µs | p95 µs | p99 µs | build (s) |
|---|---|---|---|---|---|---|
| flat | {} | 1.0000 | 3864 | 5026 | 5277 | 0.0 |
| ivf | {"nlist": 2048, "nprobe": 8} | 0.3053 | 86 | 95 | 103 | 8.2 |
| ivf | {"nlist": 2048, "nprobe": 32} | 0.6086 | 165 | 184 | 197 | 8.2 |
| ivf | {"nlist": 2048, "nprobe": 64} | 0.7637 | 293 | 378 | 393 | 8.2 |
| hnsw | {"M": 32, "ef_search": 64} | 0.4800 | 312 | 396 | 445 | 20.2 |
| hnsw | {"M": 32, "ef_search": 128} | 0.6353 | 464 | 614 | 680 | 20.2 |
| hnsw | {"M": 32, "ef_search": 256} | 0.7792 | 770 | 982 | 1059 | 20.2 |

Latency: single query, single thread, includes Python call overhead.
