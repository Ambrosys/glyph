"""Tests for module control.gp.algorithms."""

import re
import operator

import pytest
import numpy as np
import dill

from glyph import gp
from glyph import assessment
from glyph.utils.numeric import rms, hill_climb


class SingleConstIndividual(gp.AExpressionTree):
    """An individual class."""
    pset = gp.sympy_primitive_set(categories=['algebraic', 'exponential', 'neg'], arguments=['x_0'], constants=['c'])
    marker = "sympy"


class TwoConstIndividual(gp.AExpressionTree):
    """An individual class."""
    pset = gp.sympy_primitive_set(categories=['algebraic', 'exponential', 'neg'], arguments=['x_0'], constants=['c_0', 'c_1'])
    marker = "sympy"


class UnlimitedConstants(gp.AExpressionTree):
    pset = gp.numpy_primitive_set(1, categories=('algebraic', 'trigonometric', 'exponential', 'symc'))
    marker = "symc"


class Any(int):
    def __init__(self):
        self = 1.0

    def __eq__(self, other):
        """Equality is always true."""
        return True

    def __add__(self, other):
        return 0


def assert_all_close(a, b, r_tol=1e-6):
    """Custom all_close to account for comparisons to Any()"""
    def tupleize(x):
        try:
            iter(x)
        except:
            x = x,
        return x
    a = tupleize(a)
    b = tupleize(b)
    assert all([abs(p+operator.neg(q)) <= r_tol for p, q in zip(a, b)])


const_opt_agreement_cases = [
    (SingleConstIndividual, 'x_0', lambda x: x, np.linspace(0, 100, 100), 1.0, 1),
    (SingleConstIndividual, 'Mul(c, x_0)', lambda x: x, np.linspace(0, 100, 100), 1.0, 1),
    (SingleConstIndividual, 'Mul(c, Neg(x_0))', lambda x: 1.5 * x, np.linspace(0, 100, 100), -1.5, 1),
    (SingleConstIndividual, 'Add(c, Mul(c, x_0))', lambda x: 8.0 + 8.0 * x, np.linspace(0, 100, 100), 8.0, 1),
    (SingleConstIndividual, 'Add(c, x_0)', lambda x: 5.5 + 5.5 * x, np.linspace(-100, 100, 200), 5.5, 1),
    (TwoConstIndividual, 'x_0', lambda x: x, np.linspace(0, 100, 100), 1.0, 2),
    (TwoConstIndividual, 'Mul(c_1, x_0)', lambda x: 2.0 * x, np.linspace(0, 100, 100), (Any(), 2.0), 2),
    (TwoConstIndividual, 'Add(Mul(c_0, x_0), c_1)', lambda x: 4.1 * x + 2.3, np.linspace(0, 100, 100), (4.1, 2.3), 2),
    (UnlimitedConstants, 'Mul(Symc, x_0)', lambda x: 1.5 * x, np.linspace(0, 100, 100), 1.5, 1),
    (UnlimitedConstants, 'Mul(Symc, Add(x_0, Symc)', lambda x: x + 2.0, np.linspace(0, 100, 100), (1.0, 2.0), 2),
    (UnlimitedConstants, 'x_0', lambda x: x, np.linspace(0, 100, 100), (), 0),
]


class Measure:
    def __init__(self, target, x):
        self.x = x
        self.target = target

    def __call__(self, individual, *consts):
        phenotype = gp.sympy_phenotype if individual.marker == "sympy" else gp.numpy_phenotype
        func = phenotype(individual)
        return func(self.x, *consts) - self.target(self.x)


@pytest.mark.parametrize('case', const_opt_agreement_cases)
def test_const_opt_leastsq(case):
    individual_class, expr, target, x, desired, _ = case
    ind = individual_class.from_string(expr)
    m = Measure(target, x)
    popt, _ = assessment.const_opt_leastsq(m, ind)
    assert_all_close(desired, popt)


@pytest.mark.parametrize('case', const_opt_agreement_cases)
def test_const_opt_scalar(case):
    individual_class, expr, target, x, desired, _ = case
    ind = individual_class.from_string(expr)
    m = Measure(target, x)

    def error(individual, *consts):
        residuals = m(individual, *consts)
        return rms(residuals)

    popt, _ = assessment.const_opt_scalar(error, ind)
    assert_all_close(desired, popt)

def test_hill_climb():
    rng = np.random.RandomState(seed=1742)
    case = const_opt_agreement_cases[3]
    optiones = {"directions": 200, "maxiter": 2000, "target": 0.2, 'rng': rng}

    individual_class, expr, target, x, desired, _ = case
    ind = individual_class.from_string(expr)
    m = Measure(target, x)

    def error(individual, *consts):
        residuals = m(individual, *consts)
        return rms(residuals)

    popt, rms_opt = assessment.const_opt_scalar(error, ind, method=hill_climb, options=optiones)
    assert rms_opt <= optiones["target"]
    assert_all_close(desired, popt, r_tol=0.1)



@pytest.mark.parametrize('case', const_opt_agreement_cases)
def test_default_constants(case):
    individual_class, expr, _, _, _, n_consts = case
    ind = individual_class.from_string(expr)
    np.testing.assert_allclose(actual=assessment.default_constants(ind), desired=np.ones(n_consts), rtol=0)


def test_numpy_phenotype():
    expr = "Add(Mul(Symc, x_1), Mul(Symc, x_0)))"
    pset = gp.numpy_primitive_set(2, categories=('algebraic', 'trigonometric', 'exponential', 'symc'))
    ind = type("ind", (gp.AExpressionTree, ), dict(pset=pset)).from_string(expr)
    f = gp.numpy_phenotype(ind)
    assert f(0, 0) == 0
    assert f(1, 1) == 2
    assert f(1, 2, 2, 1) == 5
    assert f(1, 2, c_0=2, c_1=1) == 5
    with pytest.raises(TypeError):
        f(1, 1, 1, 1, 1)


@pytest.mark.parametrize('case', filter(lambda x: x[0] is UnlimitedConstants, const_opt_agreement_cases))
def test__get_index(case):
    individual_class, expr, _, _, _, n_consts = case
    ind = individual_class.from_string(expr)
    c = "Symc"
    splitter = lambda s: [s_.strip() for s_ in re.split("\(|,|\,|\)", s)]
    index = [i for i, node in enumerate(splitter(expr)) if node == c]
    assert index == gp.individual._get_index(ind, c)


def test_pickle_assessment_runner():
    arunner = assessment.AAssessmentRunner()
    brunner = dill.loads(dill.dumps(arunner))
    assert type(arunner.parallel_factory) == type(brunner.parallel_factory)
    del arunner.parallel_factory
    del brunner.parallel_factory
    assert arunner.__dict__ == brunner.__dict__


def test_replace_nan():
    assert assessment.replace_nan(np.nan) == np.infty
    assert assessment.replace_nan([np.nan]) == [np.infty]


def test__tt_flatten():
    tpl = ((1,), (3,))
    assert assessment._tt_flatten(tpl) == (1, 3)


def test_tuple_wrap():
    f = lambda x: x
    f = assessment.tuple_wrap(f)
    f1 = f(1)
    assert isinstance(f1, tuple)
    f = assessment.tuple_wrap(f)
    assert f1 == f(1)
    g = assessment.tuple_wrap(lambda x: [x])
    assert isinstance(g(1), tuple)
