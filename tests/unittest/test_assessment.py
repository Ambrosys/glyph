"""Tests for module control.gp.algorithms."""

import pytest
import numpy

from glyph import gp
from glyph import assessment


class SingleConstIndividual(gp.AExpressionTree):
    """An individual class."""
    pset = gp.sympy_primitive_set(categories=['algebraic', 'exponential'], arguments=['x_0'], constants=['c'])


class TwoConstIndividual(gp.AExpressionTree):
    """An individual class."""
    pset = gp.sympy_primitive_set(categories=['algebraic', 'exponential'], arguments=['x_0'], constants=['c_0', 'c_1'])

# TODO(jg): Add more test expressions.
const_opt_agreement_cases = [
    (SingleConstIndividual, 'x_0', lambda x: x, numpy.linspace(0, 100, 100), 1.0),
    (SingleConstIndividual, 'Mul(c, x_0)', lambda x: x, numpy.linspace(0, 100, 100), 1.0),
    (SingleConstIndividual, 'Mul(c, Neg(x_0))', lambda x: 1.5 * x, numpy.linspace(0, 100, 100), -1.5),
    (SingleConstIndividual, 'Add(c, Mul(c, x_0))', lambda x: 8.0 + 8.0 * x, numpy.linspace(0, 100, 100), 8.0),
    (SingleConstIndividual, 'Add(c, x_0)', lambda x: 5.5 + 5.5 * x, numpy.linspace(-100, 100, 200), 5.5),
    # (SingleConstIndividual, 'exp(c, x_0)', lambda x: numpy.exp(3.0 * x), numpy.linspace(-100, 100, 200), 3.0),
    (TwoConstIndividual, 'x_0', lambda x: x, numpy.linspace(0, 100, 100), 1.0),
    (TwoConstIndividual, 'Mul(c_1, x_0)', lambda x: 2.0 * x, numpy.linspace(0, 100, 100), (1.0, 2.0)),
    (TwoConstIndividual, 'Add(Mul(c_0, x_0), c_1)', lambda x: 4.1 * x + 2.3, numpy.linspace(0, 100, 100), (4.1, 2.3)),
    # (TwoConstIndividual, 'Mul(c_0, exp(c_1, Neg(x_0))', lambda x: 3.4 * numpy.exp(-1.5 * x), numpy.linspace(0, 100, 100), (3.4, 1.5)),
    # (TwoConstIndividual, 'Add(c_0, exp(c_1, x_0))', lambda x: 8.0 + numpy.exp(1.4 * x), numpy.linspace(0, 100, 100), (8.0, 1.4)),
]


class Measure:
    def __init__(self, target, x):
        self.x = x
        self.target = target

    def __call__(self, individual, *fargs):
        func = gp.sympy_phenotype(individual)
        return func(self.x, *fargs) - self.target(self.x)


@pytest.mark.parametrize('case', const_opt_agreement_cases)
def test_const_opt_scalar_agreement(case):
    individual_class, expr, target, x, desired = case
    ind = individual_class.from_string(expr)
    p0 = numpy.ones(len(ind.pset.constants))
    m = Measure(target, x)
    popt, _ = assessment.const_opt_leastsq(m, ind, p0)
    numpy.testing.assert_allclose(actual=popt, desired=desired, rtol=1e-6)


