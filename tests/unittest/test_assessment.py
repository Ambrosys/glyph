"""Tests for module control.gp.algorithms."""

import re

import pytest
import numpy
import dill

from glyph import gp
from glyph import assessment


class SingleConstIndividual(gp.AExpressionTree):
    """An individual class."""
    pset = gp.sympy_primitive_set(categories=['algebraic', 'exponential'], arguments=['x_0'], constants=['c'])
    marker = "sympy"


class TwoConstIndividual(gp.AExpressionTree):
    """An individual class."""
    pset = gp.sympy_primitive_set(categories=['algebraic', 'exponential'], arguments=['x_0'], constants=['c_0', 'c_1'])
    marker = "sympy"


class UnlimitedConstants(gp.AExpressionTree):
    pset = gp.numpy_primitive_set(1, categories=('algebraic', 'trigonometric', 'symc'))
    marker = "symc"


# TODO(jg): Add more test expressions.
const_opt_agreement_cases = [
    (SingleConstIndividual, 'x_0', lambda x: x, numpy.linspace(0, 100, 100), 1.0, 1),
    (SingleConstIndividual, 'Mul(c, x_0)', lambda x: x, numpy.linspace(0, 100, 100), 1.0, 1),
    (SingleConstIndividual, 'Mul(c, Neg(x_0))', lambda x: 1.5 * x, numpy.linspace(0, 100, 100), -1.5, 1),
    (SingleConstIndividual, 'Add(c, Mul(c, x_0))', lambda x: 8.0 + 8.0 * x, numpy.linspace(0, 100, 100), 8.0, 1),
    (SingleConstIndividual, 'Add(c, x_0)', lambda x: 5.5 + 5.5 * x, numpy.linspace(-100, 100, 200), 5.5, 1),
    (TwoConstIndividual, 'x_0', lambda x: x, numpy.linspace(0, 100, 100), 1.0, 2),
    (TwoConstIndividual, 'Mul(c_1, x_0)', lambda x: 2.0 * x, numpy.linspace(0, 100, 100), (1.0, 2.0), 2),
    (TwoConstIndividual, 'Add(Mul(c_0, x_0), c_1)', lambda x: 4.1 * x + 2.3, numpy.linspace(0, 100, 100), (4.1, 2.3), 2),
    # (TwoConstIndividual, 'Mul(c_0, exp(c_1, Neg(x_0))', lambda x: 3.4 * numpy.exp(-1.5 * x), numpy.linspace(0, 100, 100), (3.4, 1.5)),
    # (TwoConstIndividual, 'Add(c_0, exp(c_1, x_0))', lambda x: 8.0 + numpy.exp(1.4 * x), numpy.linspace(0, 100, 100), (8.0, 1.4)),
    (UnlimitedConstants, 'Mul(Symc, x_0)', lambda x: 1.5 * x, numpy.linspace(0, 100, 100), 1.5, 1),
]


class Measure:
    def __init__(self, target, x):
        self.x = x
        self.target = target

    def __call__(self, individual, *fargs):
        phenotype = gp.sympy_phenotype if individual.marker == "sympy" else gp.numpy_phenotype
        func = phenotype(individual)
        return func(self.x, *fargs) - self.target(self.x)


@pytest.mark.parametrize('case', const_opt_agreement_cases)
def test_const_opt_scalar_agreement(case):
    individual_class, expr, target, x, desired, _ = case
    ind = individual_class.from_string(expr)
    m = Measure(target, x)
    popt, _ = assessment.const_opt_leastsq(m, ind)
    numpy.testing.assert_allclose(actual=popt, desired=desired, rtol=1e-6)


@pytest.mark.parametrize('case', const_opt_agreement_cases)
def test_default_constants(case):
    individual_class, expr, _, _, _, n_consts = case
    ind = individual_class.from_string(expr)
    numpy.testing.assert_allclose(actual=assessment.default_constants(ind), desired=numpy.ones(n_consts), rtol=0)


@pytest.mark.parametrize('case', filter(lambda x: x[0] is UnlimitedConstants, const_opt_agreement_cases))
def test__get_index(case):
    individual_class, expr, _, _, _, n_consts = case
    ind = individual_class.from_string(expr)
    c = "Symc"
    index = [i for i, node in enumerate(re.split("\(|,|\,|\)", expr)) if node == c]
    assert index == gp.individual._get_index(ind, c)


def test_pickle_assessment_runner():
    arunner = assessment.AAssessmentRunner()
    brunner = dill.loads(dill.dumps(arunner))
    assert type(arunner.parallel_factory) == type(brunner.parallel_factory)
    del arunner.parallel_factory
    del brunner.parallel_factory
    assert arunner.__dict__ == brunner.__dict__
