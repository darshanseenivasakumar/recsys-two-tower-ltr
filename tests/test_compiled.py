"""The compiled (lleaves) ranker: bit-identical to LightGBM, cached safely, never trusted
blindly. A small LambdaRank model with the served model's awkward parts (a categorical
feature, NaN-aware splits) keeps compilation to a few seconds."""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

pytest.importorskip("lleaves")

from recsys.serving.compiled import (  # noqa: E402
    cache_path,
    check_rows,
    compile_model,
    verified_compiled,
)


def _model(path: Path, seed: int) -> tuple[lgb.Booster, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, groups = 3000, 100
    x = rng.normal(size=(n, 5))
    x[:, 0] = rng.integers(0, 6, n)  # categorical code, like user_occupation
    x[rng.random(n) < 0.3, 1] = np.nan  # missing, like bpr_score for users BPR never saw
    y = (x[:, 2] + (x[:, 0] == 3) + np.nan_to_num(x[:, 1]) > 0.5).astype(int)
    data = lgb.Dataset(x, y, group=[n // groups] * groups, categorical_feature=[0])
    params = {"objective": "lambdarank", "num_leaves": 15, "verbose": -1, "seed": seed}
    booster = lgb.train(params, data, num_boost_round=40)
    booster.save_model(str(path))
    return lgb.Booster(model_file=str(path)), x


def test_compiled_scores_are_bit_identical_to_lightgbm(tmp_path: Path) -> None:
    booster, x = _model(tmp_path / "m.txt", 0)
    rows = check_rows(x)
    rows[::7, 0] = 99.0  # category never seen in training
    predict = verified_compiled(tmp_path / "m.txt", tmp_path / "cache", rows, booster)
    assert predict is not None
    np.testing.assert_array_equal(predict(rows), booster.predict(rows, num_threads=1))


def test_cache_is_reused_and_keyed_by_model_content(tmp_path: Path) -> None:
    _model(tmp_path / "a.txt", 0)
    _model(tmp_path / "b.txt", 1)
    first = compile_model(tmp_path / "a.txt", tmp_path / "cache")
    stamp = first.stat().st_mtime_ns
    assert compile_model(tmp_path / "a.txt", tmp_path / "cache") == first  # no recompile
    assert first.stat().st_mtime_ns == stamp
    assert cache_path(tmp_path / "b.txt", tmp_path / "cache") != first
    assert not list((tmp_path / "cache").glob("*.tmp"))  # temp file renamed into place


def test_a_binary_that_disagrees_with_lightgbm_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """lleaves never checks a cache for staleness. Plant model A's binary under model B's
    cache name: loading must notice the scores differ and fall back to LightGBM."""
    _model(tmp_path / "a.txt", 0)
    booster_b, x = _model(tmp_path / "b.txt", 1)
    binary_a = compile_model(tmp_path / "a.txt", tmp_path / "cache")
    binary_a.rename(cache_path(tmp_path / "b.txt", tmp_path / "cache"))
    with caplog.at_level(logging.ERROR):
        got = verified_compiled(tmp_path / "b.txt", tmp_path / "cache", x, booster_b)
    assert got is None
    assert "differs from LightGBM" in caplog.text


def test_missing_cache_without_compile_falls_back(tmp_path: Path) -> None:
    booster, x = _model(tmp_path / "m.txt", 0)
    got = verified_compiled(
        tmp_path / "m.txt", tmp_path / "cache", x, booster, compile_if_missing=False
    )
    assert got is None
