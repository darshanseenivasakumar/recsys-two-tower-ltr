"""The drift reference: what the ranker's input looked like when it was trained.

Sampled exactly like serving samples its feature log: `per_query` uniformly random
candidate rows per personalized query (>= 1 positive before the decision; serving only
builds features for those), drawn from the ranker's training queries in w1 + w2, cutoff
and session queries alike, i.e. the distribution the ranker was actually fit on.
Comparing like with like matters: if serving logged the top-ranked rows but the reference
were uniform, PSI would flag the sampling difference as "drift".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from recsys.config import load_data_config
from recsys.ranking.candidates import within_cap

REFERENCE_FILE = "monitoring_reference.parquet"


def sample_per_query(query: np.ndarray, per_query: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices: up to `per_query` uniformly random rows from each query."""
    frame = pd.DataFrame({"q": query, "r": rng.random(len(query))})
    frame = frame.sort_values(["q", "r"], kind="stable")
    keep = frame.groupby("q").cumcount() < per_query
    return np.sort(frame.index[keep].to_numpy())


def build_reference(
    data_config: str,
    ranker_config: str,
    per_query: int = 10,
    seed: int = 0,
    cap: int | None = None,
) -> pd.DataFrame:
    """`cap`: the serving score cap; serving only builds (and logs) features for capped
    candidates, so the reference must come from the same subset or PSI flags the rank
    features for a sampling difference."""
    from recsys.pipelines.ranker import WindowTable  # training-side import

    with open(ranker_config, encoding="utf-8") as f:
        rcfg = yaml.safe_load(f)
    data_cfg = load_data_config(data_config)
    out_dir = Path(rcfg["artifacts_dir"]) / data_cfg.dataset
    features = json.loads((out_dir / "ranker_features.json").read_text(encoding="utf-8"))[
        "features"
    ]
    rng = np.random.default_rng(seed)
    parts = []
    for name in ("w1", "w2"):
        table = WindowTable.load(out_dir / "windows" / name)
        personalized = table.queries["personalized"].to_numpy(bool)[table.rows]
        idx = np.flatnonzero(personalized & within_cap(table.features, cap))
        picked = idx[sample_per_query(table.rows[idx], per_query, rng)]
        part = table.features.iloc[picked][features].reset_index(drop=True)
        part.insert(0, "_window", name)
        part.insert(1, "_at_cutoff", table.queries["at_cutoff"].to_numpy(bool)[table.rows[picked]])
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
