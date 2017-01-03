import pytest

from glyph.cli.glyph_remote import *

@pytest.fixture(scope="module")
def primitives():
    prims = dict(f=2, x=0, k=-1)
    return prims


def test_build_pset_gp(primitives):
    pset = build_pset_gp(primitives)
    assert len(pset.terminals[object]) == 2
    assert pset.constants == {"k"}
