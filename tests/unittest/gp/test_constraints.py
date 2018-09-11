from functools import partial
import tempfile

import pytest

from glyph.gp.individual import *
from glyph.gp.constraints import *
from glyph.gp.individual import ANDimTree, add_sc, sc_qout
from glyph.gp.breeding import nd_mutation


class Tree(AExpressionTree):
    pset = numpy_primitive_set(1, categories=('algebraic', 'trigonometric', 'symc'))
    marker = "symc"


class SympyTree(AExpressionTree):
    pset = sympy_primitive_set(categories=['algebraic', 'exponential'], arguments=['x_0'], constants=['c_0'])
    marker = "sympy"

cases = (
    ("Sub(x_0, x_0)", True, dict(zero=True, constant=True, infty=True), Tree),
    ("Sub(x_0, x_0)", False, dict(zero=False, constant=False, infty=True), Tree),
    ("Div(x_0, Sub(x_0, x_0))", True, dict(zero=False, constant=True, infty=True), Tree),
    ("Div(x_0, Sub(x_0, x_0))", False, dict(zero=False, constant=True, infty=False), Tree),
    ("-1.0", True, dict(zero=True, constant=True, infty=True), Tree),
    ("1.0", False, dict(zero=True, constant=False, infty=True), Tree),
    ("SC(x_0, x_0)", True, dict(zero=True, constant=True, infty=True), Tree),
    ("Add(Symc, Symc)", True, dict(zero=True, constant=True, infty=True), Tree),
    ("Add(c_0, c_0)", True, dict(zero=True, constant=True, infty=True), SympyTree),
    ("Mul(x_0, Div(Mul(Div(x_0, x_0), Div(x_0, x_0)), Div(Add(x_0, x_0), Div(x_0, x_0))))", True, dict(zero=True, constant=True, infty=True), Tree)
)


@pytest.mark.parametrize("case", cases)
def test_NonFiniteExpression(case):
    expr, res, settings, cls = case
    cls.pset = add_sc(cls.pset, sc_qout)
    ns = NonFiniteExpression(**settings)
    ind = cls.from_string(expr)
    assert (ind in ns) == res


def mock(*inds, ret=None):
    return [ret] * max(len(inds), 1)


@pytest.mark.parametrize("i", range(3))  # create = 0, mutate = 1, mate = 2
def test_constraint_decorator(i, NumpyIndividual):

    ind = NumpyIndividual.from_string("Sub(x_0, x_0)")
    this_mock = partial(mock, ret=ind)

    ns = NonFiniteExpression()
    assert ind in ns

    [this_mock] = constrain([this_mock], ns)

    if i == 0:
        with pytest.raises(UserWarning):
            this_mock()
    else:

        other_ind, *rest = this_mock(*[ind]*i)
        assert ind == other_ind


def test_constraint_in_nd(NumpyIndividual):

    class NDTree(ANDimTree):
        base = NumpyIndividual

    ind = NumpyIndividual.from_string("Sub(x_0, x_0)")
    this_mock = partial(mock, ret=ind)
    ns = NonFiniteExpression()
    [this_mock] = constrain([this_mock], ns)
    mate = partial(nd_mutation, mut1d=this_mock)

    nd_ind = NDTree([ind]*2)

    new_nd_ind, *rest = mate(nd_ind)

    for c, d in zip(nd_ind, new_nd_ind):
        assert c == d


def test_pretest():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write("chi = lambda *args: True")
    constraint = PreTest(tmp.name)
    assert 1 in constraint
