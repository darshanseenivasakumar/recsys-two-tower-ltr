"""Turn raw MovieLens tables into clean, canonical tables saved as Parquet.

Output (under data/processed/<dataset>/):
    interactions.parquet  user_id, item_id, user_idx, item_idx, rating, timestamp, is_positive
    items.parquet         item_id, item_idx, title, year, genres (list[str])
    users.parquet         user_id, user_idx [, gender, age, occupation]

Deliberately NOT done here: anything that uses interaction statistics (k-core filtering,
popularity, per-user averages). Those must be computed *after* the temporal split, from
the training portion only, or they leak future information into training.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.data.datasets import RawTables
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
    YEAR,
)

logger = logging.getLogger(__name__)

_YEAR_AT_END = re.compile(r"\((\d{4})\)\s*$")
_NO_GENRES = "(no genres listed)"  # ML-25M's placeholder


@dataclass(frozen=True)
class ProcessedTables:
    interactions: pd.DataFrame
    items: pd.DataFrame
    users: pd.DataFrame


def parse_year(title: str) -> int | None:
    """Extract the release year from a title like 'Toy Story (1995)'.

    Anchored at the end because titles can contain other parentheses, e.g.
    'City of Lost Children, The (Cité des enfants perdus, La) (1995)'.
    """
    match = _YEAR_AT_END.search(title)
    return int(match.group(1)) if match else None


def parse_genres(raw: str | float) -> list[str]:
    """'Animation|Children's|Comedy' -> ['Animation', "Children's", 'Comedy']."""
    if not isinstance(raw, str) or raw.strip() in ("", _NO_GENRES):
        return []
    return raw.split("|")


def deduplicate_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent interaction per (user, item) pair.

    A re-rating replaces the earlier opinion. Keeping both would put the same pair in
    train AND test under a temporal split, which inflates metrics.
    """
    before = len(interactions)
    deduped = interactions.sort_values(TIMESTAMP, kind="stable").drop_duplicates(
        subset=[USER_ID, ITEM_ID], keep="last"
    )
    if dropped := before - len(deduped):
        logger.info("Dropped %d duplicate (user, item) interactions", dropped)
    return deduped


def positive_mask(ratings: npt.ArrayLike, threshold: float) -> npt.NDArray[np.bool_]:
    """THE implicit-feedback rule, shared by preprocessing and the serving API."""
    return np.asarray(ratings, dtype=np.float64) >= threshold


def add_implicit_label(interactions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add IS_POSITIVE = rating >= threshold. Raw ratings are kept for ranking features."""
    out = interactions.copy()
    out[IS_POSITIVE] = positive_mask(out[RATING].to_numpy(), threshold)
    return out


def add_contiguous_index(df: pd.DataFrame, id_col: str, idx_col: str) -> pd.DataFrame:
    """Sort by raw ID and assign a dense 0..N-1 index (what embedding tables use)."""
    out = df.sort_values(id_col).reset_index(drop=True)
    if out[id_col].duplicated().any():
        raise ValueError(f"Duplicate {id_col} values; cannot build an index")
    out[idx_col] = np.arange(len(out), dtype=np.int32)
    return out


def preprocess(raw: RawTables, positive_threshold: float) -> ProcessedTables:
    # Items: the index covers the whole catalog (movies file), including never-rated
    # movies. The catalog is known metadata, not future behavior, so this does not leak.
    items = raw.items.copy()
    items[YEAR] = items[TITLE].map(parse_year).astype("Int16")
    items[GENRES] = items[GENRES].map(parse_genres)
    items = add_contiguous_index(items, ITEM_ID, ITEM_IDX)

    users = add_contiguous_index(raw.users, USER_ID, USER_IDX)

    interactions = deduplicate_interactions(raw.interactions)
    known_items = interactions[ITEM_ID].isin(items[ITEM_ID])
    known_users = interactions[USER_ID].isin(users[USER_ID])
    if unknown := int((~(known_items & known_users)).sum()):
        logger.warning("Dropping %d interactions with IDs missing from the catalog", unknown)
        interactions = interactions[known_items & known_users]

    interactions = interactions.merge(users[[USER_ID, USER_IDX]], on=USER_ID).merge(
        items[[ITEM_ID, ITEM_IDX]], on=ITEM_ID
    )
    interactions = add_implicit_label(interactions, positive_threshold)

    # Global chronological order with a deterministic tie-break: ML-1M has many
    # interactions sharing the exact same second, and splits must be reproducible.
    interactions = interactions.sort_values(
        [TIMESTAMP, USER_ID, ITEM_ID], kind="stable"
    ).reset_index(drop=True)
    interactions = interactions[
        [USER_ID, ITEM_ID, USER_IDX, ITEM_IDX, RATING, TIMESTAMP, IS_POSITIVE]
    ]
    return ProcessedTables(interactions=interactions, items=items, users=users)


def save_processed(tables: ProcessedTables, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables.interactions.to_parquet(out_dir / "interactions.parquet", index=False)
    tables.items.to_parquet(out_dir / "items.parquet", index=False)
    tables.users.to_parquet(out_dir / "users.parquet", index=False)
    logger.info("Wrote processed tables to %s", out_dir)


def load_processed(in_dir: Path) -> ProcessedTables:
    return ProcessedTables(
        interactions=pd.read_parquet(in_dir / "interactions.parquet"),
        items=pd.read_parquet(in_dir / "items.parquet"),
        users=pd.read_parquet(in_dir / "users.parquet"),
    )
