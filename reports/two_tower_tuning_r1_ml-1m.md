# Two-tower retrieval: ml-1m

## Validation (train -> val), best epoch per config

| overrides | best epoch | val warm recall@200 | val warm NDCG@10 | val Coverage@100 | val ARP@100 | recall@200 at last epoch | fit (s) |
|---|---|---|---|---|---|---|---|
| base | 6 | 0.4288 | 0.0880 | 0.7056 | 1010 | 0.3844 | 154 |
| weight_decay=0.1 | 5 | 0.4347 | 0.0929 | 0.6340 | 1052 | 0.3825 | 155 |
| weight_decay=1.0 | 2 | 0.4579 | 0.1137 | 0.5565 | 1099 | 0.3647 | 153 |
| dropout=0.2 | 2 | 0.4159 | 0.1078 | 0.5285 | 1057 | 0.3791 | 180 |
| use_user_id=False | 9 | 0.4268 | 0.0922 | 0.6402 | 1061 | 0.4111 | 167 |
| logq_alpha=1.25 | 5 | 0.4647 | 0.1089 | 0.5305 | 1122 | 0.4174 | 149 |
| logq_alpha=1.5 | 7 | 0.4781 | 0.1125 | 0.4914 | 1136 | 0.4618 | 222 |
| batch_size=4096, lr=0.002 | 9 | 0.4426 | 0.0932 | 0.6899 | 1017 | 0.3703 | 479 |
| lr=0.0003 | 15 | 0.4343 | 0.0882 | 0.6819 | 1038 | 0.4343 | 194 |
