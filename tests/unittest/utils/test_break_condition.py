import pytest

from glyph.utils.break_condition import *


@pytest.mark.parametrize("ttl", [0, 1])
def test_SoftTimeOut(ttl):
    sttl = SoftTimeOut(ttl)

    assert sttl.alive
    time.sleep(ttl + 1)

    assert bool(ttl) != bool(sttl())
