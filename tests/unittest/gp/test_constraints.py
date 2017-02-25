from functools import partial

import pytest

from glyph.gp.constraints import *

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


def mutate_mock(ind, cls):
    """Allways return ind in Nullspace"""
    expr = "Sub(x_0, x_0)"
    return cls.from_string(expr)


def test_constraint_decorator(NumpyIndividual):

    ind = NumpyIndividual.from_string("Sub(x_0, x_0)")
    mutate = partial(mutate_mock, cls=NumpyIndividual)

    ns = NullSpace()
    assert ind in ns

    [mutate] = apply_constraints([mutate], build_constraints(ns))

    with pytest.raises(RuntimeWarning):
        mutate(ind)
