from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from recsys.data.datasets import read_ml_1m
from recsys.data.download import safe_extract
from recsys.data.schema import AGE, GENDER, ITEM_ID, RATING, TITLE, USER_ID


def test_read_ml_1m_parses_dat_format(tmp_path: Path) -> None:
    # Real ML-1M quirks: '::' separator, latin-1 accents, quotes inside titles.
    (tmp_path / "ratings.dat").write_bytes(b"1::1193::5::978300760\n2::661::3::978302109\n")
    (tmp_path / "movies.dat").write_bytes(
        "1193::One Flew Over the Cuckoo's Nest (1975)::Drama\n"
        '661::"Cité des enfants perdus, La" (1995)::Adventure|Sci-Fi\n'.encode("latin-1")
    )
    (tmp_path / "users.dat").write_bytes(b"1::F::1::10::48067\n2::M::56::16::70072-1234\n")

    raw = read_ml_1m(tmp_path)

    assert raw.interactions[RATING].tolist() == [5.0, 3.0]
    assert raw.items.loc[raw.items[ITEM_ID] == 661, TITLE].item().startswith('"Cité')
    assert raw.users[GENDER].tolist() == ["F", "M"]
    assert raw.users[AGE].tolist() == [1, 56]
    assert str(raw.interactions[USER_ID].dtype) == "int32"


def test_safe_extract_rejects_zip_slip(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "pwned")
    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()
