"""Feature drift: Population Stability Index (PSI) of serving features vs training.

    PSI = sum_b (cur_b - ref_b) * ln(cur_b / ref_b)

over bins b fixed on the REFERENCE distribution (training data): deciles of each numeric
feature, or its categories when it has few distinct values. Missing values get their own
bin, so a feature that starts arriving as NaN is drift too. Empty bins are floored at
1e-4 so the log stays finite. Conventional reading: < 0.1 stable, 0.1-0.25 moderate,
> 0.25 major shift.

PSI is symmetric-ish, unitless and comparable across features, which is why it is the
usual first alarm; it does not say whether the shift hurts accuracy (the replay's online
hit rate does that).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

PSI_MODERATE = 0.1
PSI_MAJOR = 0.25
EPS = 1e-4


@dataclass(frozen=True)
class Binning:
    edges: npt.NDArray[np.float64] | None  # numeric: interior cut points
    categories: npt.NDArray[np.float64] | None  # categorical: known values

    @property
    def n_bins(self) -> int:
        # value bins + one "unseen category" bin (categorical) + one missing-value bin
        if self.categories is not None:
            return len(self.categories) + 2
        assert self.edges is not None
        return len(self.edges) + 2

    @classmethod
    def fit(cls, reference: npt.ArrayLike, bins: int = 10, max_categories: int = 12) -> Binning:
        x = np.asarray(reference, dtype=np.float64)
        finite = x[np.isfinite(x)]
        values = np.unique(finite)
        if values.size <= max_categories:
            return cls(edges=None, categories=values)
        cuts = np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1])
        return cls(edges=np.unique(cuts), categories=None)

    def assign(self, values: npt.ArrayLike) -> npt.NDArray[np.int64]:
        x = np.asarray(values, dtype=np.float64)
        missing = ~np.isfinite(x)
        if self.categories is not None:
            cats = self.categories
            pos = np.searchsorted(cats, x)
            known = np.zeros(len(x), dtype=bool)
            inside = pos < cats.size
            known[inside] = cats[pos[inside]] == x[inside]
            codes = np.where(known, pos, cats.size)  # values never seen in training: own bin
        else:
            assert self.edges is not None
            codes = np.searchsorted(self.edges, x, side="right")
        return np.where(missing, self.n_bins - 1, codes).astype(np.int64)

    def proportions(self, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
        codes = self.assign(values)
        counts = np.bincount(codes, minlength=self.n_bins).astype(np.float64)
        return np.maximum(counts / max(len(codes), 1), EPS)


def psi(reference: npt.ArrayLike, current: npt.ArrayLike, bins: int = 10) -> float:
    binning = Binning.fit(reference, bins)
    ref, cur = binning.proportions(reference), binning.proportions(current)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def status(value: float) -> str:
    if value >= PSI_MAJOR:
        return "major"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def drift_table(
    reference: pd.DataFrame, current: pd.DataFrame, features: Sequence[str], bins: int = 10
) -> pd.DataFrame:
    """PSI per feature, worst first, with means and missing rates for context."""
    rows = []
    for name in features:
        ref = reference[name].to_numpy(dtype=np.float64)
        cur = current[name].to_numpy(dtype=np.float64)
        value = psi(ref, cur, bins)
        rows.append(
            {
                "feature": name,
                "psi": value,
                "status": status(value),
                "ref_mean": float(np.nanmean(ref)) if np.isfinite(ref).any() else np.nan,
                "cur_mean": float(np.nanmean(cur)) if np.isfinite(cur).any() else np.nan,
                "ref_missing": float(np.mean(~np.isfinite(ref))),
                "cur_missing": float(np.mean(~np.isfinite(cur))),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def noise_floor(
    reference: pd.DataFrame, features: Sequence[str], n_current: int, seed: int = 0
) -> float:
    """Max PSI between the reference and a same-size random subsample of itself: how large
    PSI gets from sampling noise alone at this sample size."""
    rng = np.random.default_rng(seed)
    sample = reference.iloc[
        rng.choice(len(reference), size=min(n_current, len(reference)), replace=False)
    ]
    return float(drift_table(reference, sample, features)["psi"].max())
