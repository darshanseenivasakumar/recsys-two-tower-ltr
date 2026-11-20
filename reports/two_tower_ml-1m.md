# Two-tower retrieval: ml-1m

## Validation (train -> val), best epoch per config

| overrides | best epoch | val warm recall@200 | val warm NDCG@10 | val Coverage@100 | val ARP@100 | recall@200 at last epoch | fit (s) |
|---|---|---|---|---|---|---|---|
| logq_alpha=3.0 | 9 | 0.4918 | 0.1424 | 0.2792 | 1275 | 0.4725 | 203 |
| logq_alpha=4.0 | 9 | 0.4880 | 0.1430 | 0.2264 | 1307 | 0.4738 | 196 |
| logq_correction=False, temperature=0.1 | 20 | 0.1724 | 0.0260 | 0.9995 | 287 | 0.1724 | 197 |

## Test (train+val -> test), exact search, all users

| Model | Recall@100 | Recall@200 | NDCG@10 | Recall@10 | Coverage@100 | LongTail@100 | ARP@100 |
|---|---|---|---|---|---|---|---|
| most_popular | 0.2700 | 0.4060 | 0.2190 | 0.0666 | 0.1455 | 0.0052 | 1322 |
| item_knn | 0.2896 | 0.4222 | 0.2486 | 0.0637 | 0.2073 | 0.0068 | 1264 |
| als | 0.3265 | 0.4669 | 0.2568 | 0.0713 | 0.2004 | 0.0059 | 1254 |
| bpr | 0.3396 | 0.4954 | 0.2624 | 0.0756 | 0.4494 | 0.0840 | 1094 |
| two_tower (logq_alpha=3.0) | 0.3216 | 0.4700 | 0.2264 | 0.0671 | 0.3124 | 0.0204 | 1225 |
| two_tower (logq_correction=False, temperature=0.1) | 0.0921 | 0.1608 | 0.0392 | 0.0126 | 1.0000 | 0.8078 | 226 |

## FAISS on the real catalog (3883 items, test users, k=200)

| index | params | Recall@200 | overlap@200 vs exact | p50 µs | p95 µs | p99 µs |
|---|---|---|---|---|---|---|
| flat | {} | 0.4700 | 1.0000 | 103 | 125 | 151 |
| ivf | {"nlist": 64, "nprobe": 1} | 0.1534 | 0.2379 | 14 | 15 | 22 |
| ivf | {"nlist": 64, "nprobe": 2} | 0.2162 | 0.3520 | 19 | 22 | 30 |
| ivf | {"nlist": 64, "nprobe": 4} | 0.3021 | 0.5197 | 26 | 30 | 37 |
| ivf | {"nlist": 64, "nprobe": 8} | 0.4156 | 0.7312 | 52 | 60 | 75 |
| ivf | {"nlist": 64, "nprobe": 16} | 0.4664 | 0.9235 | 44 | 62 | 72 |
| hnsw | {"M": 32, "ef_search": 16} | 0.2514 | 0.3575 | 35 | 47 | 62 |
| hnsw | {"M": 32, "ef_search": 32} | 0.3512 | 0.5379 | 55 | 72 | 98 |
| hnsw | {"M": 32, "ef_search": 64} | 0.4360 | 0.7613 | 84 | 99 | 112 |
| hnsw | {"M": 32, "ef_search": 128} | 0.4632 | 0.9349 | 134 | 191 | 233 |
| hnsw | {"M": 32, "ef_search": 256} | 0.4693 | 0.9927 | 249 | 426 | 702 |

## FAISS on a synthetic 300,000-item catalog (ANN vs exact only)

| index | params | overlap@200 vs exact | p50 µs | p95 µs | p99 µs | build (s) |
|---|---|---|---|---|---|---|
| flat | {} | 1.0000 | 4901 | 5688 | 6330 | 0.0 |
| ivf | {"nlist": 2048, "nprobe": 8} | 0.5775 | 106 | 134 | 148 | 14.0 |
| ivf | {"nlist": 2048, "nprobe": 32} | 0.8881 | 182 | 241 | 263 | 14.0 |
| ivf | {"nlist": 2048, "nprobe": 64} | 0.9647 | 268 | 402 | 446 | 14.0 |
| hnsw | {"M": 32, "ef_search": 64} | 0.7576 | 279 | 499 | 1572 | 26.1 |
| hnsw | {"M": 32, "ef_search": 128} | 0.8808 | 390 | 586 | 694 | 26.1 |
| hnsw | {"M": 32, "ef_search": 256} | 0.9539 | 658 | 949 | 1081 | 26.1 |

Latency: single query, single thread, includes Python call overhead.
