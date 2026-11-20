"""Recommendation models and a registry to build them by name from configs."""

from __future__ import annotations

from typing import Any

from recsys.models.base import ColdStartFallback, Recommender


def build_model(name: str, params: dict[str, Any], cold_start_fallback: bool = True) -> Recommender:
    """Instantiate a model from its registry name and hyperparameters.

    Personalized models are wrapped so users without positive history get most-popular.
    Imports are local so that, e.g., serving code doesn't import torch unless it needs BPR.
    """
    from recsys.models.popular import MostPopular, RandomRecommender

    model: Recommender
    if name == "random":
        return RandomRecommender(**params)
    if name == "most_popular":
        return MostPopular(**params)
    if name == "item_knn":
        from recsys.models.item_knn import ItemKNN

        model = ItemKNN(**params)
    elif name == "als":
        from recsys.models.als import ALS

        model = ALS(**params)
    elif name == "bpr":
        from recsys.models.bpr import BPRMF

        model = BPRMF(**params)
    else:
        raise ValueError(f"Unknown model {name!r}")
    return ColdStartFallback(model, MostPopular()) if cold_start_fallback else model
