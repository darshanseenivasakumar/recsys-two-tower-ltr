# Two-tower retrieval: ml-1m

## Validation (train -> val), best epoch per config

| overrides | best epoch | val warm recall@200 | val warm NDCG@10 | val Coverage@100 | val ARP@100 | recall@200 at last epoch | fit (s) |
|---|---|---|---|---|---|---|---|
| logq_alpha=2.0 | 6 | 0.4866 | 0.1338 | 0.3747 | 1229 | 0.4650 | 154 |
| logq_alpha=3.0 | 9 | 0.4918 | 0.1424 | 0.2792 | 1275 | 0.4866 | 150 |
| logq_alpha=1.5, weight_decay=1.0 | 9 | 0.4781 | 0.1340 | 0.4355 | 1136 | 0.4093 | 150 |
| logq_alpha=2.0, weight_decay=1.0 | 12 | 0.4836 | 0.1250 | 0.3618 | 1183 | 0.4332 | 198 |
| logq_alpha=1.5, use_user_id=False | 7 | 0.4751 | 0.1105 | 0.4620 | 1161 | 0.4503 | 215 |
| logq_alpha=2.0, use_user_id=False | 9 | 0.4856 | 0.1336 | 0.3461 | 1240 | 0.4492 | 215 |
| logq_alpha=2.0, weight_decay=1.0, use_user_id=False | 4 | 0.4799 | 0.1348 | 0.3096 | 1210 | 0.4214 | 216 |
