from functools import partial

import pytest

from glyph.gp.constraints import *
from glyph.gp.individual import ANDimTree
from glyph.gp.breeding import nd_mutation

cases = (
    ("Sub(x_0, x_0)", True, dict(zero=True, constant=True, infty=True)),
    ("Sub(x_0, x_0)", False, dict(zero=False, constant=True, infty=True)),
    ("Div(x_0, Sub(x_0, x_0))", True, dict(zero=False, constant=True, infty=True)),
    ("Div(x_0, Sub(x_0, x_0))", False, dict(zero=False, constant=True, infty=False)),
    ("1.0", True, dict(zero=True, constant=True, infty=True)),
    ("1.0", False, dict(zero=True, constant=False, infty=True)),
)

@pytest.mark.parametrize("case", cases)
def test_nullspace(case, NumpyIndividual):
    expr, res, settings = case
    ns = NullSpace(**settings)
    ind = NumpyIndividual.from_string(expr)
    assert (ind in ns) == res


def mock(*inds, ret=None):
    return ret,


@pytest.mark.parametrize("i", range(3)) # create = 0, mutate = 1, mate = 2
def test_constraint_decorator(i, NumpyIndividual):

    ind = NumpyIndividual.from_string("Sub(x_0, x_0)")
    this_mock = partial(mock, ret=ind)

    ns = NullSpace()
    assert ind in ns

    [this_mock] = apply_constraints([this_mock], build_constraints(ns))

    with pytest.raises(RuntimeWarning):
        this_mock(*[ind]*i)


def test_constraint_in_nd(NumpyIndividual):

    class NDTree(ANDimTree):
        base = NumpyIndividual

    ind = NumpyIndividual.from_string("Sub(x_0, x_0)")
    this_mock = partial(mock, ret=ind)
    ns = NullSpace()
    [this_mock] = apply_constraints([this_mock], build_constraints(ns))
    mate = partial(nd_mutation, mut1d=this_mock)

    nd_ind = NDTree([ind]*2)
    print(nd_ind)
    with pytest.raises(RuntimeWarning):
        mate(nd_ind)
