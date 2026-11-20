"""Typed configuration loaded from YAML.

YAML is the source of truth for experiment settings; these dataclasses give us
autocomplete, type checking and validation at load time instead of KeyErrors deep
inside a training run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

SplitStrategy = Literal["global_temporal", "leave_last_n"]


@dataclass(frozen=True)
class PathsConfig:
    """Where data lives. Relative paths resolve against the current working directory."""

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    splits_dir: Path = Path("data/splits")


@dataclass(frozen=True)
class SplitConfig:
    """How to partition interactions into train / validation / test."""

    strategy: SplitStrategy = "global_temporal"
    val_frac: float = 0.1
    test_frac: float = 0.1
    n_val: int = 1
    n_test: int = 1

    def __post_init__(self) -> None:
        if self.strategy not in get_args(SplitStrategy):
            allowed = get_args(SplitStrategy)
            raise ValueError(f"Unknown split strategy {self.strategy!r}; expected one of {allowed}")
        if not (0 < self.val_frac < 1 and 0 < self.test_frac < 1):
            raise ValueError("val_frac and test_frac must each be in (0, 1)")
        if self.val_frac + self.test_frac >= 1:
            raise ValueError("val_frac + test_frac must be < 1 so that training data remains")
        if self.n_val < 0 or self.n_test < 1:
            raise ValueError("Need n_val >= 0 and n_test >= 1")


@dataclass(frozen=True)
class DataConfig:
    """Top-level data pipeline config."""

    dataset: str
    positive_threshold: float = 4.0
    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    @property
    def raw_path(self) -> Path:
        return self.paths.raw_dir / self.dataset

    @property
    def processed_path(self) -> Path:
        return self.paths.processed_dir / self.dataset

    @property
    def split_path(self) -> Path:
        return self.paths.splits_dir / self.dataset / self.split.strategy


def load_data_config(path: str | Path) -> DataConfig:
    """Load and validate a data config YAML file."""
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    paths = PathsConfig(**{key: Path(value) for key, value in raw.get("paths", {}).items()})
    split = SplitConfig(**raw.get("split", {}))
    return DataConfig(
        dataset=raw["dataset"],
        positive_threshold=float(raw.get("positive_threshold", 4.0)),
        paths=paths,
        split=split,
    )
