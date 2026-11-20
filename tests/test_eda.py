from __future__ import annotations

import pytest

from recsys.data.eda import gini, head_share


def test_gini_even_and_concentrated() -> None:
    assert gini([5, 5, 5, 5]) == pytest.approx(0.0)
    # sorted [0,0,0,10]: 2*(4*10)/(4*10) - 5/4 = 0.75 (max possible for n=4 is 0.75)
    assert gini([10, 0, 0, 0]) == pytest.approx(0.75)
    assert gini([]) == 0.0


def test_head_share() -> None:
    shares = head_share([5, 50, 10, 30, 5], fracs=(0.2, 0.4))
    assert shares[0.2] == pytest.approx(0.5)  # top 1 of 5 items: 50/100
    assert shares[0.4] == pytest.approx(0.8)  # top 2: (50+30)/100
