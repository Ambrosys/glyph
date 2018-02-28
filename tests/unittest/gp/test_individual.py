import inspect

import dill
import pytest
import sympy

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
    assert str(simplify_this(ind)) in desired # test in not equality for appveyor +inf edge case


def test_simplify_this_random_state():
    individual_class, expr, desired = simplify_cases[0]
    ind = individual_class.from_string(expr)

    import random
    random.seed(42)
    s = random.getstate()

    simplify_this(ind)

    assert random.getstate() == s


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


get_len_case = (
    ("x_0", 1),
    ("exp(x_0)", 2),
    ("Add(x_0, x_0)", 3),
    ("Add(x_0, Add(x_0, x_0))", 5),
)

@pytest.mark.parametrize("case", get_len_case)
def test_get_len(case):
    expr, x = case
    assert StructConst.get_len(expr) == x


def test_struct_const_format():
    f = lambda x, y: x + y
    sc = StructConst(f)

    left = "Add(x0, x0)"
    right = "y0"

    assert sc.format(left, right) == str(4)


@pytest.fixture
def sc_ind():
    this_pset = numpy_primitive_set(1)
    f = lambda x, y: 0
    this_pset = add_sc(this_pset, f)
    import operator
    this_pset.addPrimitive(operator.neg, 1, "Neg")
    class Ind(AExpressionTree):
        pset = this_pset
    return Ind


@pytest.mark.parametrize("case", get_len_case)
def test_struct_const_repr(case, sc_ind):
    subexpr, _ = case
    expr = "Add(x_0, SC({subexpr}, {subexpr}))".format(subexpr=subexpr)
    ind = sc_ind.from_string(expr)

    func = numpy_phenotype(ind)
    assert func(1) == 1


def test_simplify_this_struct_const(sc_ind):
    expr = "SC(0.0, Add(x_0, Neg(x_0)))"
    ind = sc_ind.from_string(expr)
    assert str(ind) == str(simplify_this(ind))


child_tree_cases = (
    ("x_0", []),
    ("Add(x_0, x_0)", ["x_0", "x_0"]),
)

@pytest.mark.parametrize("case", child_tree_cases)
def test_child_trees(case):
    expr, res = case
    ind = Tree.from_string(expr)
    assert list(child_trees(ind)) == [Tree.from_string(r) for r in res]


simplify_consts_cases = (
    ("Add(x_0, Add(Symc, Symc))", "Add(x_0, Symc)"),
    ("Symc", "Symc"),
    ("sin(cos(Symc)", "Symc"),
    ("Div(x_0, cos(Symc))", "Div(x_0, Symc)"),
    ("Div(x_0, Add(Symc, Symc))", "Div(x_0, Symc)"),
    ("Add(x_0, Symc)", "Add(x_0, Symc)"),
    ("Div(Symc, Add(x_0, Symc))", "Div(Symc, Add(x_0, Symc))"),
)

@pytest.mark.parametrize("case", simplify_consts_cases)
def test_simplify_constant(case):
    expr, res = case
    ind = Tree.from_string(expr)
    assert str(simplify_constant(ind)) == res



pprint_cases = (
    ("c", "1", ["c"], [1], 0),
    ("c1", "1", ["c1"], [1], 0),
    ("subtract(c, c)", "subtract(1, 1)", ["c"], [1], 0),
    ("add(c, c)", "add(1, c)", ["c"], [1], 1),
    ("add(c, c)", "add(1, 2)", ["c", "c"], [1, 2], 1),
    ("c+x", "1+x", ["c"], [1], 0),
    ("c + x", "1 + x", ["c"], [1], 0),
    ("a * x", "1 * x", ["a"], [1], 0),
)

@pytest.mark.parametrize("case", pprint_cases)
def test_pretty_print(case):
    expr, res, constants, values, count = case
    assert res == pretty_print(expr, constants, values, count=count)


def test_pprint_simplify():
    expr = "Add(x_0, Symc)"
    ind = Tree.from_string(expr)
    simple_expr = str(simplify_this(ind))
    res = pretty_print(simple_expr, ind.pset.constants, [1])
    assert res == "1 + x_0"


constant_normal_form_cases =(
    ("f(a+b)", "c"),
    ("2*a", "c"),
    ("2*b + c + d", "c"),
    ("f(y_0)", "f(y_0)"),
    ("y_0 + f(y_0 + 3*b)", "y_0 + f(c + y_0)"),
    ("c**3", "c"),
    ("sin(c)", "c"),
    ("y_0**2", "y_0**2")
)

@pytest.mark.parametrize("case", constant_normal_form_cases)
def test__constant_normal_form(case):
    from glyph.gp.individual import _constant_normal_form
    expr, res = case
    assert res == repr(_constant_normal_form(sympy.sympify(expr), variables=[sympy.Symbol("y_0")]))
