"""Tests for module control.gp.algorithms."""

import operator
import re
import sys

import dill
import numpy as np
import pytest
import scipy.integrate

from glyph import assessment
from glyph import gp
from glyph.utils.numeric import hill_climb, rms

SingleConstIndividual = gp.Individual(
    pset=gp.sympy_primitive_set(
        categories=["algebraic", "exponential", "neg", "sqrt"], arguments=["x_0"], constants=["c"]
    ),
    name="SingleConstIndividual",
    marker="sympy",
)

TwoConstIndividual = gp.Individual(
    pset=gp.sympy_primitive_set(
        categories=["algebraic", "exponential", "neg"], arguments=["x_0"], constants=["c_0", "c_1"]
    ),
    name="TwoConstIndividual",
    marker="sympy",
)

UnlimitedConstants = gp.Individual(
    pset=gp.numpy_primitive_set(1, categories=("algebraic", "trigonometric", "exponential", "symc")),
    name="UnlimitedConstants",
    marker="symc",
)


class Any(int):
    def __init__(self):
        self = 1.0

    def __eq__(self, other):
        """Equality is always true."""
        return True

    def __add__(self, other):
        return 0


def tupleize(x):
    try:
        iter(x)
    except:
        x = (x,)
    return x


def assert_all_close(a, b, r_tol=1e-6):
    """Custom all_close to account for comparisons to Any()"""
    a = tupleize(a)
    b = tupleize(b)
    assert all([abs(p + operator.neg(q)) <= r_tol for p, q in zip(a, b)])


const_opt_agreement_cases = [
    (SingleConstIndividual, "x_0", lambda x: x, np.linspace(0, 100, 100), 1.0, 1),
    (SingleConstIndividual, "Mul(c, x_0)", lambda x: x, np.linspace(0, 100, 100), 1.0, 1),
    (SingleConstIndividual, "Mul(c, Neg(x_0))", lambda x: 1.5 * x, np.linspace(0, 100, 100), -1.5, 1),
    (SingleConstIndividual, "Add(c, Mul(c, x_0))", lambda x: 8.0 + 8.0 * x, np.linspace(0, 100, 100), 8.0, 1),
    (SingleConstIndividual, "Add(c, x_0)", lambda x: 5.5 + 5.5 * x, np.linspace(-100, 100, 200), 5.5, 1),
    (TwoConstIndividual, "x_0", lambda x: x, np.linspace(0, 100, 100), 1.0, 2),
    (TwoConstIndividual, "Mul(c_1, x_0)", lambda x: 2.0 * x, np.linspace(0, 100, 100), (Any(), 2.0), 2),
    (
        TwoConstIndividual,
        "Add(Mul(c_0, x_0), c_1)",
        lambda x: 4.1 * x + 2.3,
        np.linspace(0, 100, 100),
        (4.1, 2.3),
        2,
    ),
    (UnlimitedConstants, "Mul(Symc, x_0)", lambda x: 1.5 * x, np.linspace(0, 100, 100), 1.5, 1),
    (
        UnlimitedConstants,
        "Mul(Symc, Add(x_0, Symc)",
        lambda x: x + 2.0,
        np.linspace(0, 100, 100),
        (1.0, 2.0),
        2,
    ),
    (UnlimitedConstants, "x_0", lambda x: x, np.linspace(0, 100, 100), (), 0),
    (
        SingleConstIndividual,
        "sqrt(Neg(c))",
        lambda x: x,
        np.linspace(0, 100, 100),
        Any(),
        1,
    ),  # raises exception
]


class Measure:
    def __init__(self, target, x):
        self.x = x
        self.target = target

    def __call__(self, individual, *consts):
        phenotype = gp.sympy_phenotype if individual.marker == "sympy" else gp.numpy_phenotype
        func = phenotype(individual)
        return func(self.x, *consts) - self.target(self.x)


@pytest.mark.parametrize("case", const_opt_agreement_cases)
def test_const_opt_leastsq(case):
    individual_class, expr, target, x, desired, _ = case
    ind = individual_class.from_string(expr)
    m = Measure(target, x)
    popt, _ = assessment.const_opt_leastsq(m, ind)
    assert_all_close(desired, popt)


@pytest.mark.parametrize("case", const_opt_agreement_cases)
def test_const_opt_scalar(case):
    individual_class, expr, target, x, desired, _ = case
    ind = individual_class.from_string(expr)
    m = Measure(target, x)

    def error(individual, *consts):
        residuals = m(individual, *consts)
        return rms(residuals)

    popt, _ = assessment.const_opt_scalar(error, ind, method="Powell")
    assert_all_close(desired, popt)


def test_const_opt_scalar_functional():
    def integral(ind, *const):
        f = gp.sympy_phenotype(ind)

        def f_square(x, *const):
            return f(x, *const) ** 2

        s, *_ = scipy.integrate.quad(f_square, 0, 1, args=tupleize(const))
        return s

    expr = "Mul(c, x_0)"
    ind = SingleConstIndividual.from_string(expr)

    popt, _ = assessment.const_opt_scalar(integral, ind)
    assert_all_close(popt, [0])


def test_hill_climb():
    rng = np.random.RandomState(seed=1742)
    case = const_opt_agreement_cases[3]
    optiones = {"directions": 200, "maxfev": 2000, "target": 0.2, "rng": rng}

    individual_class, expr, target, x, desired, _ = case
    ind = individual_class.from_string(expr)
    m = Measure(target, x)

    def error(individual, *consts):
        residuals = m(individual, *consts)
        return rms(residuals)

    popt, rms_opt = assessment.const_opt_scalar(error, ind, method=hill_climb, options=optiones)
    assert rms_opt <= optiones["target"]
    assert_all_close(desired, popt, r_tol=0.1)


@pytest.mark.parametrize("case", const_opt_agreement_cases)
def test_default_constants(case):
    individual_class, expr, _, _, _, n_consts = case
    ind = individual_class.from_string(expr)
    np.testing.assert_allclose(actual=assessment.default_constants(ind), desired=np.ones(n_consts), rtol=0)


@pytest.mark.parametrize("case", filter(lambda x: x[0] is UnlimitedConstants, const_opt_agreement_cases))
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


@pytest.mark.parametrize("x", [[np.nan], np.array([np.nan]), np.nan])
def test_replace_nan(x):
    x_clean = assessment.replace_nan(x)
    assert isinstance(x_clean, type(x))
    try:
        assert np.nan not in x_clean
    except:
        assert not np.isnan(x_clean)


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


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_max_fitness_on_timeout():

    f = lambda: 1

    def g():
        import time

        time.sleep(10)
        return f()

    decorator = assessment.max_fitness_on_timeout(2, 1)

    assert decorator(f)() == 1
    assert decorator(g)() == 2


ec_cases = (
    ("x_0", 1),
    ("Add(exp(x_0), x_0)", 8),
)


@pytest.mark.parametrize("expr, res", ec_cases)
def test_expressional_complexity(NumpyIndividual, expr, res):
    assert assessment.expressional_complexity(NumpyIndividual.from_string(expr)) == res
