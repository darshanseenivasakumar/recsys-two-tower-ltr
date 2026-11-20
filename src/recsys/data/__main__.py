"""Data pipeline CLI.

python -m recsys.data download   --config configs/data/ml-1m.yaml
python -m recsys.data preprocess --config configs/data/ml-1m.yaml
python -m recsys.data split      --config configs/data/ml-1m.yaml
python -m recsys.data all        --config configs/data/ml-1m.yaml
"""

from __future__ import annotations

import argparse
import json
import logging

from recsys.config import DataConfig, load_data_config
from recsys.data.datasets import get_dataset_spec
from recsys.data.download import download_dataset
from recsys.data.preprocess import load_processed, preprocess, save_processed
from recsys.data.split import make_split, save_split


def run_download(cfg: DataConfig, force: bool = False) -> None:
    download_dataset(get_dataset_spec(cfg.dataset), cfg.paths.raw_dir, force=force)


def run_preprocess(cfg: DataConfig) -> None:
    spec = get_dataset_spec(cfg.dataset)
    raw = spec.reader(cfg.raw_path / spec.extracted_dir)
    tables = preprocess(raw, cfg.positive_threshold)
    save_processed(tables, cfg.processed_path)


def run_split(cfg: DataConfig) -> None:
    interactions = load_processed(cfg.processed_path).interactions
    summary = save_split(make_split(interactions, cfg.split), cfg.split_path)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m recsys.data")
    parser.add_argument("command", choices=["download", "preprocess", "split", "all"])
    parser.add_argument("--config", required=True, help="Path to a configs/data/*.yaml file")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_data_config(args.config)

    if args.command in ("download", "all"):
        run_download(cfg, force=args.force)
    if args.command in ("preprocess", "all"):
        run_preprocess(cfg)
    if args.command in ("split", "all"):
        run_split(cfg)


if __name__ == "__main__":
    main()
