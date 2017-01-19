import inspect
from glyph.gp.individual import numpy_phenotype, numpy_primitive_set, AExpressionTree, Symc


class Tree(AExpressionTree):
    pset = numpy_primitive_set(1, categories=('algebraic', 'symc'))


def test_hash(IndividualClass):
    ind = IndividualClass.create_population(1)[0]
    pop = [ind, ind]
    assert len(set(pop)) == 1


def test_reproducibility(IndividualClass):
    import random

    seed = 1234567890
    random.seed(seed)
    population_1 = IndividualClass.create_population(1000)
    random.seed(seed)
    population_2 = IndividualClass.create_population(1000)
    assert population_1 == population_2


def test_symc_from_string():
    expr = "Symc"
    ind = Tree.from_string(expr)
    assert ind[0].name == Symc.__name__


def test_numpy_phenotype():
    expr = "Add(x0, Symc)"
    ind = Tree.from_string(expr)
    f = numpy_phenotype(ind)
    assert f(1) == 2

    signature = inspect.signature(f)
    assert "x0" in signature.parameters
    assert signature.parameters["c0"].default == 1
