"""Load test for the recommender API.

    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 -t 2m --csv reports/loadtest/u50

Traffic mix per simulated user: 90% GET /recommend, 10% POST /interaction. User ids are
drawn from the whole user table, so the ~mix of warm and cold users matches the bundle
(cold users exercise the popularity fallback). Each POST changes that user's next
/recommend, as real traffic would; reseed Redis after a run to restore the snapshot.

FastHttpUser (geventhttpclient), not HttpUser (requests): `requests` sends a POST's headers
and body as separate TCP segments, and Nagle + delayed ACK then adds a constant ~40 ms
that is pure client artifact (we measured /interaction at ~45 ms p50 for one RPUSH).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pandas as pd
from locust import FastHttpUser, between, task

BUNDLE = Path(os.environ.get("RECSYS_BUNDLE_DIR", "artifacts/serving/ml-1m"))
USER_IDS = pd.read_parquet(BUNDLE / "users.parquet")["user_id"].astype(str).tolist()
ITEM_IDS = pd.read_parquet(BUNDLE / "catalog.parquet")["item_id"].astype(str).tolist()


class RecommenderUser(FastHttpUser):
    wait_time = between(0.2, 1.0)  # think time between a user's requests

    @task(9)
    def recommend(self) -> None:
        self.client.get(
            "/recommend",
            params={"user_id": random.choice(USER_IDS), "k": 10},
            name="/recommend",
        )

    @task(1)
    def interact(self) -> None:
        self.client.post(
            "/interaction",
            json={
                "user_id": random.choice(USER_IDS),
                "item_id": random.choice(ITEM_IDS),
                "rating": random.randint(1, 5),
            },
            name="/interaction",
        )
