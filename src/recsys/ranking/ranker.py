"""LightGBM LambdaRank over candidate lists (one query group = one user's candidates).

LambdaRank: for every (relevant, irrelevant) pair in a group, the gradient pushing them
apart is the RankNet gradient scaled by |delta NDCG| from swapping their positions. Pairs
near the top of the list therefore dominate training, matching the NDCG@k objective.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.features.ranking import CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankingTable:
    """Feature rows grouped by query. Rows of one query must be contiguous."""

    features: pd.DataFrame
    labels: npt.NDArray[np.int64]
    query: npt.NDArray[np.int64]  # query id per row (contiguous runs)

    def group_sizes(self) -> npt.NDArray[np.int64]:
        if len(self.query) == 0:
            return np.empty(0, dtype=np.int64)
        change = np.flatnonzero(np.diff(self.query)) + 1
        if len(np.unique(self.query)) != len(change) + 1:
            raise ValueError("rows of each query must be contiguous")
        return np.diff(np.concatenate([[0], change, [len(self.query)]])).astype(np.int64)

    def with_positive_queries(self) -> RankingTable:
        """Drop queries with no positive: they carry no LambdaRank gradient, and LightGBM
        scores their NDCG as 1, which would inflate early-stopping validation."""
        has_pos = pd.Series(self.labels).groupby(self.query).transform("max").to_numpy() > 0
        return RankingTable(
            self.features[has_pos].reset_index(drop=True), self.labels[has_pos], self.query[has_pos]
        )


@dataclass
class LambdaRankRanker:
    params: dict[str, Any] = field(default_factory=dict)
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    booster_: lgb.Booster = field(init=False, repr=False)
    feature_names_: list[str] = field(init=False, default_factory=list)

    def _dataset(self, table: RankingTable, reference: lgb.Dataset | None = None) -> lgb.Dataset:
        categorical = [c for c in CATEGORICAL_FEATURES if c in table.features.columns]
        return lgb.Dataset(
            table.features,
            label=table.labels,
            group=table.group_sizes(),
            categorical_feature=categorical,
            reference=reference,
            free_raw_data=False,
        )

    def fit(self, train: RankingTable, valid: RankingTable) -> Self:
        train, valid = train.with_positive_queries(), valid.with_positive_queries()
        self.feature_names_ = list(train.features.columns)
        train_set = self._dataset(train)
        valid_set = self._dataset(valid, reference=train_set)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [10],
            "verbosity": -1,
            **self.params,
        }
        self.booster_ = lgb.train(
            params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(100),
            ],
        )
        logger.info(
            "LambdaRank best iteration %d, valid ndcg@10 %.4f",
            self.booster_.best_iteration,
            self.booster_.best_score["valid"]["ndcg@10"],
        )
        return self

    def predict(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray(
            self.booster_.predict(
                features[self.feature_names_], num_iteration=self.booster_.best_iteration
            ),
            dtype=np.float64,
        )

    def feature_importance(self) -> pd.DataFrame:
        gain = self.booster_.feature_importance("gain", iteration=self.booster_.best_iteration)
        splits = self.booster_.feature_importance("split", iteration=self.booster_.best_iteration)
        table = pd.DataFrame({"feature": self.feature_names_, "gain": gain, "splits": splits})
        table["gain_share"] = table["gain"] / max(table["gain"].sum(), 1e-12)
        return table.sort_values("gain", ascending=False).reset_index(drop=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster_.save_model(str(path), num_iteration=self.booster_.best_iteration)
