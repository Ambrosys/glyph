import pytest
import operator
import deap
import glyph.gp as gp


@pytest.fixture(params=gp.all_mutations)
def mutate_factory(request):
    return request.param


@pytest.fixture(params=gp.all_crossover)
def mate_factory(request):
    return request.param


def test_mutate_reproducibility(mutate_factory, SympyIndividual):
    import random
    seed = 1234567890
    num_iterations = 100

    mutate = mutate_factory(SympyIndividual.pset)
    static_limit_decorator = deap.gp.staticLimit(key=operator.attrgetter("height"), max_value=24)
    static_limit_decorator(mutate)

    random.seed(seed)
    ind_1 = SympyIndividual.create_population(1)[0]
    for _ in range(num_iterations):
        ind_1, = mutate(ind_1)

    random.seed(seed)
    ind_2 = SympyIndividual.create_population(1)[0]
    for _ in range(num_iterations):
        ind_2, = mutate(ind_2)

    assert ind_1 == ind_2


def test_mate_reproducibility(mate_factory, SympyIndividual):
    import random
    seed = 1234567890
    num_iterations = 100

    mate = mate_factory()
    static_limit_decorator = deap.gp.staticLimit(key=operator.attrgetter("height"), max_value=24)
    static_limit_decorator(mate)

    random.seed(seed)
    ind_1 = SympyIndividual.create_population(2)
    for _ in range(num_iterations):
        ind_1 = mate(*ind_1)

    random.seed(seed)
    ind_2 = SympyIndividual.create_population(2)
    for _ in range(num_iterations):
        ind_2 = mate(*ind_2)

    assert ind_1 == ind_2
