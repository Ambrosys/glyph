"""Tests for module control.gp.algorithms."""

import pytest
import deap.tools
import numpy as np

from glyph.utils.numeric import nrmse, silent_numpy
from glyph.gp.breeding import mutuniform, cxonepoint
from glyph.assessment import replace_nan

slow = pytest.mark.skipif(not pytest.config.getoption("--runslow"), reason="need --runslow option to run")

# TODO(jg): Use a set of fixed individuals as test population -- not create_population(), since it produces a random populations.


def mate_mock(ind1, ind2):
    """Mock for mating functions.

    :returns: (ind1, ind2) unchanged
    """
    return ind1, ind2


def mutate_mock(ind1):
    """Mock for mutation functions.

    :returns: (ind1, ) unchanged
    """
    return ind1,


def fitness_values(population):
    """Return fitness values of every indivdual in population."""
    return [ind.fitness.values for ind in population]


def set_fitnesses(population, fitness_values):
    """Set fitness values of every indivdual in population to fitness_values."""
    for ind in population:
        ind.fitness.values = fitness_values


def invalid_individuals(population):
    """Return individuals with invalid fitness values."""
    return [ind for ind in population if not ind.fitness.valid]


def test_invalid_fitness_error(AlgorithmClass, IndividualClass):
    """Test for RunError in case there are any invalid fitness values in a population."""
    pop_size = 10
    population = IndividualClass.create_population(pop_size)
    algorithm = setup_algorithm(AlgorithmClass, IndividualClass)
    with pytest.raises(RuntimeError):
        population = algorithm.evolve(population)


def test_best_is_preserved(AlgorithmClass, IndividualClass):
    """Make sure the best indidvidual is preserved (single valued fitness values)."""
    pop_size = 50
    num_generations = 10
    population = IndividualClass.create_population(pop_size)
    # Set initial fitness values and construct a best indivdual.
    set_fitnesses(population, (10.0, 10.0))
    population[0].fitness.values = (1.0, 1.0)
    best = population[0]
    # Run algorithm.
    algorithm = setup_algorithm(AlgorithmClass, IndividualClass)
    for _ in range(num_generations):
        population = algorithm.evolve(population)
        set_fitnesses(invalid_individuals(population), (10.0, 10.0))
    assert best in population


def test_paretobest_are_preserved(AlgorithmClass, IndividualClass):
    """Make sure at least some of the pareto-best indidviduals are preserved (multivalued fitness values)."""
    pop_size = 50
    num_generations = 10
    population = IndividualClass.create_population(pop_size)
    # Set initial fitness values and construct a pareto front.
    set_fitnesses(population, (10.0, 10.0))
    population[0].fitness.values = (0.0, 2.0)
    population[1].fitness.values = (1.0, 1.0)
    population[2].fitness.values = (2.0, 0.0)
    reference_pareto_front = population[0:3]
    # Run algorithm.
    algorithm = setup_algorithm(AlgorithmClass, IndividualClass)

    for _ in range(num_generations):
        population = algorithm.evolve(population)
        set_fitnesses(invalid_individuals(population), (10.0, 10.0))
    # Extract new pareto front from population and compare to constructed one.
    pareto_front = deap.tools.ParetoFront()
    pareto_front.update(population)
    assert set(fitness_values(pareto_front)) & set(fitness_values(reference_pareto_front)) != {}


@slow
@pytest.mark.skip
@pytest.mark.parametrize("select", [deap.tools.selNSGA2, deap.tools.selSPEA2, deap.tools.selBest])
def test_selection_bug_nan_fitness(select, IndividualClass):
    """This test replicates the best individual overall dropping out of the current population if randomly competing
    with "invalid" individuals, i.e. individuals with some nan fitness values
    """
    pop_size = 2
    num_generations = 10
    population = IndividualClass.create_population(pop_size)
    # Set initial fitness values and construct a pareto front.
    population[-1].fitness.values = 0, 0
    set_fitnesses(invalid_individuals(population), (float('nan'), float('nan')))
    best = population[-1]
    assert best != select(population, 1)[0]


def get_best(pop):
    return deap.tools.selBest(pop, 1)[0]

# maybe this is not a unit test anymore
@slow
def test_best_is_preserved_with_data(AlgorithmClass, NumpyIndividual):
    from sklearn.datasets import load_boston
    pop_size = 50
    num_generation = 5

    @silent_numpy
    def measure(individual):
        func = individual.compile()
        data = load_boston()
        yhat = func(*data.data.T)
        if np.isscalar(yhat):
            yhat = np.ones_like(data.target) * yhat
        fit = nrmse(data.target, yhat), len(individual)
        return replace_nan(fit)

    mate_mock = cxonepoint()
    mutate_mock = mutuniform(NumpyIndividual.pset)

    if "AgeFitness" in AlgorithmClass.__name__:
        algorithm = AlgorithmClass(mate_mock, mutate_mock, deap.tools.selNSGA2, NumpyIndividual.create_population)
    else:
        algorithm = AlgorithmClass(mate_mock, mutate_mock)
    population = NumpyIndividual.create_population(pop_size)
    best = get_best(population)
    for _ in range(num_generation):
        for ind, fitness in zip(population, map(measure, population)):
            ind.fitness.values = fitness
        population = algorithm.evolve(population)
        new_best = get_best(population)
        assert new_best.fitness >= best.fitness   # better means lower fitness in this case
        best = new_best


@slow
def test_reproducibility(AlgorithmClass, IndividualClass):
    import random

    seed = 1234567890
    num_generations = 10

    algorithm = setup_algorithm(AlgorithmClass, IndividualClass)

    initial_population = IndividualClass.create_population(20)
    set_fitnesses(invalid_individuals(initial_population), (10.0, 10.0))

    random.seed(seed)
    population_1 = algorithm.evolve(initial_population)
    for _ in range(num_generations - 1):
        set_fitnesses(invalid_individuals(population_1), (10.0, 10.0))
        population_1 = algorithm.evolve(population_1)

    random.seed(seed)
    population_2 = algorithm.evolve(initial_population)
    for _ in range(num_generations - 1):
        set_fitnesses(invalid_individuals(population_2), (10.0, 10.0))
        population_2 = algorithm.evolve(population_2)

    assert population_1 == population_2


def setup_algorithm(AlgorithmClass, IndividualClass):
    if "AgeFitness" in AlgorithmClass.__name__:
        algorithm = AlgorithmClass(mate_mock, mutate_mock, deap.tools.selNSGA2, IndividualClass.create_population)
    else:
        algorithm = AlgorithmClass(mate_mock, mutate_mock)
    return algorithm
