"""The serving bundle: every artifact the API needs, built offline from one data snapshot.

    python -m recsys.serving.bundle --data-config configs/data/ml-1m.yaml \
        --ranker-config configs/ranker.yaml

Snapshot = train + val (exactly the history the ranker was evaluated against on test), so
what the API serves is the model whose offline numbers are in reports/ranker_<dataset>.md.

Contents (artifacts/serving/<dataset>/):
  manifest.json           versions, cutoff, k per retriever, MMR lambda, feature list
  two_tower.pt / .json    user + item towers; known_users.npy; item_vectors.npy
  bpr.npz                 user/item factors + item bias, warm mask, cold-user scores
  popular.npy             cold-start ranking (most-popular, the Phase 2 window)
  features.pkl            ItemTable (stats at cutoff), ItemFeatures, UserFeatures
  ranker.txt              LightGBM LambdaRank model
  catalog.parquet         item_idx -> item_id, title, genres  (response payloads)
  users.parquet           user_idx -> user_id
  history.parquet         the snapshot's interactions, used to seed Redis
  monitoring_reference.parquet   training-time feature sample the drift monitor compares to
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import yaml

from recsys.data.schema import (
    GENRES,
    IS_POSITIVE,
    ITEM_ID,
    ITEM_IDX,
    RATING,
    TIMESTAMP,
    TITLE,
    USER_ID,
    USER_IDX,
)
from recsys.features.encoders import ItemFeatures, UserFeatures
from recsys.features.ranking import ItemTable
from recsys.models.two_tower import TwoTowerConfig, TwoTowerRecommender

logger = logging.getLogger(__name__)
BUNDLE_FORMAT = 1


@dataclass(frozen=True)
class BPRArrays:
    """BPR in plain arrays: score(u, i) = user[u] . item[i] + bias[i] (float32, as trained)."""

    user: npt.NDArray[np.float32]
    item: npt.NDArray[np.float32]
    bias: npt.NDArray[np.float32]
    warm: npt.NDArray[np.bool_]  # users with >= 1 positive in the snapshot
    cold_scores: npt.NDArray[np.float32]  # what ColdStartFallback scores everyone else with

    def has_vector(self, user_idx: int) -> bool:
        return 0 <= user_idx < len(self.warm) and bool(self.warm[user_idx])

    def scores(self, user_idx: int) -> npt.NDArray[np.float32]:
        """Dense scores for retrieval. Users without a vector get the popularity fallback
        (good candidates), but those values are counts, not dot products: the bpr_score
        FEATURE is set to NaN for them (see retrieval_features)."""
        if not self.has_vector(user_idx):
            return self.cold_scores
        u = torch.from_numpy(self.user[user_idx : user_idx + 1])
        # Same expression and dtype as BPRMF.score, so serving scores match offline ones.
        out = u @ torch.from_numpy(self.item).T + torch.from_numpy(self.bias).T
        return out.numpy()[0]


@dataclass
class ServingBundle:
    manifest: dict[str, Any]
    two_tower: TwoTowerRecommender
    bpr: BPRArrays
    popular: npt.NDArray[np.int64]
    item_table: ItemTable
    item_features: ItemFeatures
    user_features: UserFeatures
    ranker_path: Path
    catalog: pd.DataFrame
    user_index: dict[str, int]  # raw user_id -> user_idx
    item_index: dict[str, int]  # raw item_id -> item_idx

    @property
    def n_items(self) -> int:
        return len(self.catalog)

    @classmethod
    def load(cls, directory: Path) -> ServingBundle:
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["format"] != BUNDLE_FORMAT:
            raise ValueError(f"bundle format {manifest['format']} != {BUNDLE_FORMAT}")
        with open(directory / "features.pkl", "rb") as f:
            feats = pickle.load(f)  # our own artifact, written by build_bundle
        config = TwoTowerConfig(
            **json.loads((directory / "two_tower.json").read_text(encoding="utf-8"))
        )
        two_tower = TwoTowerRecommender.from_trained(
            config,
            feats["item_features"],
            feats["user_features"],
            torch.load(directory / "two_tower.pt", map_location="cpu", weights_only=True),
            np.load(directory / "known_users.npy"),
            np.load(directory / "item_vectors.npy"),
        )
        with np.load(directory / "bpr.npz") as z:
            bpr = BPRArrays(z["user"], z["item"], z["bias"], z["warm"], z["cold_scores"])
        catalog = pd.read_parquet(directory / "catalog.parquet")
        users = pd.read_parquet(directory / "users.parquet")
        return cls(
            manifest=manifest,
            two_tower=two_tower,
            bpr=bpr,
            popular=np.load(directory / "popular.npy"),
            item_table=feats["item_table"],
            item_features=feats["item_features"],
            user_features=feats["user_features"],
            ranker_path=directory / "ranker.txt",
            catalog=catalog,
            user_index=dict(
                zip(users[USER_ID].astype(str), users[USER_IDX].astype(int), strict=True)
            ),
            item_index=dict(
                zip(catalog[ITEM_ID].astype(str), catalog[ITEM_IDX].astype(int), strict=True)
            ),
        )


def build_bundle(
    data_config: str,
    ranker_config: str,
    out_dir: Path | None = None,
    ranker_file: str = "ranker.txt",
    reference_queries: str = "all",
) -> Path:
    """Fit/collect every serving artifact for the train+val snapshot.

    `ranker_file` picks the exported ranker (e.g. ranker_cutoff_only.txt for the replay
    A/B); `reference_queries` ("all" or "cutoff") must match the queries it trained on, so
    the drift monitor compares serving against that ranker's own training distribution.
    """
    from recsys.config import load_data_config
    from recsys.data.preprocess import load_processed
    from recsys.data.split import load_split
    from recsys.models import build_model
    from recsys.models.popular import MostPopular

    with open(ranker_config, encoding="utf-8") as f:
        rcfg = yaml.safe_load(f)
    data_cfg = load_data_config(data_config)
    tables = load_processed(data_cfg.processed_path)
    split = load_split(data_cfg.split_path)
    history = pd.concat([split.train, split.val], ignore_index=True)
    n_users, n_items = len(tables.users), len(tables.items)
    out = Path(out_dir or Path("artifacts") / "serving" / data_cfg.dataset)
    out.mkdir(parents=True, exist_ok=True)

    item_features = ItemFeatures.from_items(tables.items)
    user_features = UserFeatures.from_users(tables.users)
    retr = rcfg["retrievers"]

    # Two-tower: the Phase 3 export was trained on exactly this snapshot (train+val).
    tt_dir = Path(retr["tt"]["two_tower_config"]).parent
    tt_config = json.loads((tt_dir / "config.json").read_text(encoding="utf-8"))
    shutil.copy(tt_dir / "model.pt", out / "two_tower.pt")
    shutil.copy(tt_dir / "item_embeddings.npy", out / "item_vectors.npy")
    (out / "two_tower.json").write_text(json.dumps(tt_config, indent=2), encoding="utf-8")
    known = np.zeros(n_users, dtype=bool)
    known[history.loc[history[IS_POSITIVE], USER_IDX].unique()] = True
    np.save(out / "known_users.npy", known)

    # BPR: refit on the snapshot with the Phase 2 params (deterministic given the seed).
    baselines = json.loads(Path(retr["bpr"]["baselines_report"]).read_text(encoding="utf-8"))
    bpr_params = next(b["params"] for b in baselines if b["model"] == "bpr")
    start = time.perf_counter()
    bpr = build_model("bpr", bpr_params).fit(history, n_users, n_items)
    logger.info("BPR refit in %.0fs", time.perf_counter() - start)
    mf = bpr.model.model_  # type: ignore[attr-defined]
    np.savez(
        out / "bpr.npz",
        user=mf.user.weight.detach().numpy(),
        item=mf.item.weight.detach().numpy(),
        bias=mf.item_bias.weight.detach().numpy(),
        warm=bpr.is_warm_,  # type: ignore[attr-defined]
        cold_scores=bpr.fallback.score(np.array([0]))[0].astype(np.float32),  # type: ignore[attr-defined]
    )

    # Cold start: the most-popular variant selected in Phase 2.
    pop_params = next(b["params"] for b in baselines if b["model"] == "most_popular")
    pop = MostPopular(**pop_params).fit(history, n_users, n_items)
    np.save(out / "popular.npy", np.argsort(-pop.counts_, kind="stable").astype(np.int64))

    with open(out / "features.pkl", "wb") as f:
        pickle.dump(
            {
                "item_table": ItemTable.from_history(history, tables.items),
                "item_features": item_features,
                "user_features": user_features,
            },
            f,
        )
    ranker_dir = Path(rcfg["artifacts_dir"]) / data_cfg.dataset
    shutil.copy(ranker_dir / ranker_file, out / "ranker.txt")
    ranker_meta = json.loads((ranker_dir / "ranker_features.json").read_text(encoding="utf-8"))

    catalog = tables.items[[ITEM_IDX, ITEM_ID, TITLE, GENRES]].copy()
    catalog[GENRES] = catalog[GENRES].map(lambda g: "|".join(g))
    catalog.sort_values(ITEM_IDX).to_parquet(out / "catalog.parquet", index=False)
    tables.users[[USER_IDX, USER_ID]].to_parquet(out / "users.parquet", index=False)
    history[[USER_IDX, ITEM_IDX, RATING, TIMESTAMP]].to_parquet(
        out / "history.parquet", index=False
    )

    manifest = {
        "format": BUNDLE_FORMAT,
        "dataset": data_cfg.dataset,
        "version": time.strftime("%Y%m%d-%H%M%S"),
        "snapshot": "train+val",
        "ranker_file": ranker_file,
        "cutoff_unix": int(history[TIMESTAMP].max()),
        "positive_threshold": data_cfg.positive_threshold,
        "k": {"tt": int(retr["tt"]["k"]), "bpr": int(retr["bpr"]["k"])},
        "mmr": {"pool": int(rcfg["mmr"]["pool"]), "lambda": float(rcfg["mmr"]["serve_lambda"])},
        "ranker_features": ranker_meta["features"],
        "score_cap": ranker_meta.get("score_cap", {}).get(ranker_file),
        "two_tower": asdict(TwoTowerConfig(**tt_config)),
        "bpr_params": bpr_params,
        "popular_params": pop_params,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    from recsys.monitoring.reference import REFERENCE_FILE, build_reference

    reference = build_reference(data_config, ranker_config, cap=manifest["score_cap"])
    if reference_queries == "cutoff":
        reference = reference[reference["_at_cutoff"]].reset_index(drop=True)
    reference.to_parquet(out / REFERENCE_FILE, index=False)
    logger.info("serving bundle written to %s", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m recsys.serving.bundle")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--ranker-config", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ranker-file", default="ranker.txt")
    parser.add_argument("--reference-queries", choices=["all", "cutoff"], default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_bundle(
        args.data_config, args.ranker_config, args.out, args.ranker_file, args.reference_queries
    )


if __name__ == "__main__":
    main()
