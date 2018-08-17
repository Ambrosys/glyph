from glyph.observer import *


def test_get_limits():
    x = (1, 2)
    assert x == get_limits(x, factor=1.0)
