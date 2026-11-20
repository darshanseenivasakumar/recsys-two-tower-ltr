"""The LambdaRank trees compiled to native code (lleaves), verified against LightGBM.

LightGBM walks its 292 trees through a generic interpreter; lleaves turns each tree into
straight-line machine code via LLVM. On the served model that is ~5x faster per request
(2.4 -> 0.5 ms for ~110 candidates, one thread) with bit-identical scores.

Bit-identical needs one setting: lleaves by default evaluates trees in cache blocks of 34,
which sums the 292 leaf values in a different order than LightGBM, and float addition is
not associative (max difference 6e-15 on the test rows). `fblocksize = n_trees` keeps
LightGBM's order, so every score matches to the last bit and the offline/online parity
argument carries over unchanged.

The binary is specialized to the CPU it was compiled on, so it is compiled on the serving
machine (`python -m recsys.serving.compiled`, run once before the workers start) and
cached under a key of model bytes + compiler versions + host CPU. That compile is slow:
~3 min, because with blocking off the 292 trees are one function for LLVM to optimize
(33 s with the default blocking), so the cache should outlive the container (compose
keeps it in a volume). Loading checks the
compiled model against LightGBM on real feature rows before it serves anything; any
difference, a missing package or a compile error falls back to LightGBM and logs why.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import lightgbm as lgb
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

Predict = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def _host_cpu() -> str:
    import llvmlite.binding as llvm

    llvm.initialize_native_target()
    return f"{llvm.get_host_cpu_name()}|{llvm.get_host_cpu_features().flatten()}"


def cache_path(model_file: Path, cache_dir: Path) -> Path:
    """Where this model's binary lives on this machine. lleaves itself never checks a cache
    for staleness, so everything that changes the binary goes into the name."""
    h = hashlib.sha256(Path(model_file).read_bytes())
    for part in (metadata.version("lleaves"), metadata.version("llvmlite"), _host_cpu()):
        h.update(part.encode())
    return Path(cache_dir) / f"ranker-{h.hexdigest()[:20]}.bin"


def compile_model(model_file: Path, cache_dir: Path) -> Path:
    """Compile once into the cache (a no-op if it's already there). Written to a temp file
    and renamed, so concurrent workers never load a half-written binary."""
    import lleaves

    target = cache_path(model_file, cache_dir)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f".{os.getpid()}.tmp")
    model = lleaves.Model(model_file=str(model_file))
    model.compile(cache=str(tmp), fblocksize=model.num_trees())
    os.replace(tmp, target)
    return target


def _load(model_file: Path, binary: Path) -> Predict:
    import lleaves

    model = lleaves.Model(model_file=str(model_file))
    # the flags must match the ones the binary was compiled with
    model.compile(cache=str(binary), fblocksize=model.num_trees())
    # one thread: per-request blocks are ~110 rows, and workers already use every core
    return lambda x: model.predict(x, n_jobs=1)


def check_rows(rows: npt.NDArray[np.float64], seed: int = 0) -> npt.NDArray[np.float64]:
    """Real feature rows plus a copy with 15% of cells missing, so the NaN branches of every
    split (bpr_score is often NaN) are exercised too."""
    rng = np.random.default_rng(seed)
    holes = rows.copy()
    holes[rng.random(holes.shape) < 0.15] = np.nan
    return np.vstack([rows, holes])


def verified_compiled(
    model_file: Path,
    cache_dir: Path,
    rows: npt.NDArray[np.float64],
    booster: lgb.Booster,
    compile_if_missing: bool = True,
) -> Predict | None:
    """The compiled ranker if it reproduces LightGBM bit for bit on `rows`, else None."""
    try:
        binary = cache_path(model_file, cache_dir)
        if not binary.exists():
            if not compile_if_missing:
                logger.error("compiled ranker not in cache", extra={"path": str(binary)})
                return None
            compile_model(model_file, cache_dir)
        predict = _load(model_file, binary)
    except Exception:  # missing package, LLVM error, unreadable cache: serve LightGBM
        logger.exception("compiled ranker unavailable, using LightGBM")
        return None
    rows = np.ascontiguousarray(rows, dtype=np.float64)
    expected = booster.predict(rows, num_threads=1)
    got = predict(rows)
    if not np.array_equal(expected, got):
        logger.error(
            "compiled ranker differs from LightGBM, using LightGBM",
            extra={"max_abs_diff": float(np.nanmax(np.abs(expected - got))), "rows": len(rows)},
        )
        return None
    return predict


def main() -> None:
    p = argparse.ArgumentParser(description="Compile a bundle's ranker into the lleaves cache")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    out = compile_model(args.bundle / "ranker.txt", args.cache_dir)
    logger.info("compiled ranker ready: %s", out)


if __name__ == "__main__":
    main()
