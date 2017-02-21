import inspect

import dill
import pytest

from glyph.gp.individual import *


class Tree(AExpressionTree):
    pset = numpy_primitive_set(1, categories=('algebraic', 'trigonometric', 'symc'))
    marker = "symc"


class SympyTree(AExpressionTree):
    pset = sympy_primitive_set(categories=['algebraic', 'exponential'], arguments=['x_0'], constants=['c_0'])
    marker = "sympy"


class NDTree(ANDimTree):
    base = SympyTree


def test_hash(IndividualClass):
    ind = IndividualClass.create_population(1)[0]
    pop = [ind, ind]
    assert len(set(pop)) == 1

@pytest.mark.parametrize("cls", [Tree, SympyTree, NDTree])
def test_pickle(cls):
    defaults = inspect.getargspec(cls.create_population).defaults#
    defaults = len(defaults) if defaults else 0
    argcount = len(inspect.getargspec(cls.create_population).args)
    ind = cls.create_population(*[1]*(argcount-defaults-1))[0]
    assert dill.loads(dill.dumps(ind)) == ind

def test_reproducibility(IndividualClass):
    import random

    seed = 1234567890
    random.seed(seed)
    population_1 = IndividualClass.create_population(10)
    random.seed(seed)
    population_2 = IndividualClass.create_population(10)
    assert population_1 == population_2


def test_symc_from_string():
    expr = "Symc"
    ind = Tree.from_string(expr)
    assert ind[0].name == expr


phenotype_cases = [
    (Tree, "Add(x_0, Symc)"),
    (Tree, "Add(Symc, x_0)"),
    (SympyTree, "Add(c_0, x_0)"),
    (SympyTree, "Add(x_0, c_0)"),
]

@pytest.mark.parametrize('case', phenotype_cases)
def test_phenotype(case):
    individual_class, expr = case
    phenotype = sympy_phenotype if individual_class.marker == "sympy" else numpy_phenotype
    ind = individual_class.from_string(expr)
    f = phenotype(ind)
    assert f(1) == 2
    assert f(1, 2) == 3

    signature = inspect.signature(f)
    assert "x_0" in signature.parameters
    assert signature.parameters["c_0"].default == 1


simplify_cases = [
    (Tree, 'Mul(Symc, x_0)', 'Symc*x_0'),
    (Tree, 'Sub(Symc, x_0)', 'Symc - x_0'),
    (Tree, 'Add(x_0, x_0)', '2*x_0'),
    (Tree, 'Div(x_0, x_0)', '1'),
    (Tree, 'Div(x_0, 0.0)', '+inf*x_0'),
    (Tree, 'Mul(x_0, x_0)', 'x_0**2'),
    (Tree, 'Div(sin(x_0), cos(x_0))', 'tan(x_0)')
]


@pytest.mark.parametrize('case', simplify_cases)
def test_simplify_this(case):
    individual_class, expr, desired = case
    ind = individual_class.from_string(expr)
    assert str(simplify_this(ind)) == desired


nd_tree_case = (
    (["Add(x_0, x_0)", "Mul(c_0, x_0)"], [1], [2, 1]),
    (["Add(x_0, x_0)", "Mul(c_0, x_0)"], [1, 2], [2, 2]),
)

@pytest.mark.parametrize("case", nd_tree_case)
def test_nd_from_string(case):
    strs, _, _ = case
    ind = NDTree.from_string(strs)

    assert str(ind) == str(strs)

@pytest.mark.parametrize("case", nd_tree_case)
def test_nd_tree_phenotype(case):
    strs, x, res = case
    ind = NDTree.from_string(strs)
    f = nd_phenotype(ind)

    out = f(*x)
    assert np.allclose(out, res)
