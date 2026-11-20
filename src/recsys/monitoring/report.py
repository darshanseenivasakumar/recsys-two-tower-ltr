"""Monitoring summary over a window of logged traffic (used by GET /monitoring and the replay)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.features.ranking import SECONDS_PER_DAY
from recsys.monitoring.drift import drift_table, noise_floor
from recsys.monitoring.exposure import exposure_summary

MIN_ROWS_FOR_DRIFT = 200


def summarize_traffic(
    recs: pd.DataFrame,
    features: pd.DataFrame,
    reference: pd.DataFrame,
    feature_names: Sequence[str],
    item_popularity: npt.ArrayLike,
    snapshot_cutoff: float,
    top: int = 5,
) -> dict[str, Any]:
    """Health of one traffic window: volume, cold-start rate, latency, exposure, drift."""
    out: dict[str, Any] = {"n_requests": len(recs)}
    if recs.empty:
        return out
    personalized = recs["strategy"] == "personalized"
    latency = recs["latency_ms"].to_numpy()
    out |= {
        "personalized_share": float(personalized.mean()),
        "cold_start_share": float(1 - personalized.mean()),
        "latency_ms": {f"p{q}": float(np.percentile(latency, q)) for q in (50, 95, 99)},
        # how stale the batch features are at this point in (possibly simulated) time
        "days_since_snapshot": float((recs["ts"].max() - snapshot_cutoff) / SECONDS_PER_DAY),
        "exposure": exposure_summary(recs["items"].tolist(), item_popularity),
        "exposure_personalized": exposure_summary(
            recs.loc[personalized, "items"].tolist(), item_popularity
        ),
    }
    if len(features) >= MIN_ROWS_FOR_DRIFT:
        table = drift_table(reference, features, feature_names)
        out["drift"] = {
            "n_rows": len(features),
            "max_psi": float(table["psi"].max()),
            "noise_floor_psi": noise_floor(reference, feature_names, len(features)),
            "n_major": int((table["status"] == "major").sum()),
            "n_moderate": int((table["status"] == "moderate").sum()),
            "top": table.head(top).to_dict("records"),
        }
    return out
