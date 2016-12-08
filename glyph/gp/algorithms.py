import copy
import sys
import random
import itertools
import deap.tools
import deap.algorithms


def _all_valid(population):
    return all(ind.fitness.valid for ind in population)


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """Just a fixed version of deap.algorithm.varOr
    Reproduction needs cloning, so that fitness is not shared.
    Else the current implementation of AgeFitness will break.
    """
    assert (cxpb + mutpb) <= 1.0, "The sum of the crossover and mutation probabilities must be smaller or equal to 1.0."

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(toolbox.clone(random.choice(population)))

    return offspring


class MOGP(object):
    def __init__(self, mate_func, mutate_func, select):
        self.mate = mate_func
        self.mutate = mutate_func
        self.select = select
        self.clone = copy.deepcopy
        self.crossover_prob = 0.5
        self.mutation_prob = 0.2
        self.tournament_size = 2
        self._initialized = False

    def evolve(self, population):
        if not _all_valid(population):
            raise RuntimeError('Cannot evolve on invalid fitness values in population.')
        if not self._initialized:
            self._init(population)
        parents = self.select(population, self.parents_size)
        offspring = self._breed(parents)
        return parents[:] + offspring

    def _init(self, population):
        self.parents_size = len(population)
        self.offspring_size = len(population)
        self._initialized = True

    def _breed(self, parents):
        mating_pool = deap.tools.selTournament(parents, self.offspring_size, self.tournament_size)
        return varOr(mating_pool, self, self.offspring_size, self.crossover_prob, self.mutation_prob)


class NSGA2(MOGP):
    """Implementation of the NSGA-II algorithm as described in Essentials of Metaheuristics"""
    def __init__(self, mate_func, mutate_func):
        super().__init__(mate_func, mutate_func, deap.tools.selNSGA2)


class SPEA2(MOGP):
    """Implementation of the SPEA2 algorithm as described in Essentials of Metaheuristics"""
    def __init__(self, mate_func, mutate_func):
        super().__init__(mate_func, mutate_func, deap.tools.selSPEA2)


class DeapEaSimple(object):
    """Basically a copy of deap.algorithm's eaSimple algorithm."""

    def __init__(self, mate_func, mutate_func, assessment_runner=None, initial_population=()):
        self.mate = mate_func
        self.mutate = mutate_func
        self.clone = copy.deepcopy
        self.crossover_prob = 0.5
        self.mutation_prob = 0.2
        self.tournament_size = 2

        self.assessment_runner = assessment_runner
        self.initial_population = initial_population

    def evolve(self, population):
        if not _all_valid(population):
            raise RuntimeError('Cannot evolve on invalid fitness values in population.')
        offspring_size = len(population)
        parents = deap.tools.selTournament(population, offspring_size, self.tournament_size)
        offspring = deap.algorithms.varAnd(parents, self, self.crossover_prob, self.mutation_prob)  # Breeding.
        return population[:] + offspring


class AgeFitness(MOGP):
    def __init__(self, mate_func, mutate_func, select, create):
        super().__init__(mate_func, mutate_func, select)
        self.create = create
        self.num_new_blood = 1

    def _init(self, population):
        super()._init(population)
        self.offspring_size -= self.num_new_blood
        self.n_obj = len(population[0].fitness.values)

    def evolve(self, population):
        if not _all_valid(population):
            raise RuntimeError('Cannot evolve on invalid fitness values in population.')
        if not self._initialized:
            self._init(population)
        self._aging(population)
        parents = self.select(population, self.parents_size)[:]
        offspring = self._breed(parents)
        self._remove_age_from_fitness(parents + offspring)
        return parents + offspring + self.create(self.num_new_blood)

    def _remove_age_from_fitness(self, pop):
        for ind in pop:
            ind.fitness.values = ind.fitness.values[:self.n_obj]

    @staticmethod
    def _aging(population):
        for ind in population:
            ind.age = getattr(ind, "age", 0) + 1
            ind.fitness.values = *ind.fitness.values, ind.age


def make_unique_version(obj):
    uname = "U{}".format(obj.__name__)
    uobj = type(uname, (obj,), {})

    def evolve(self, population):
        return list(set(super().evolve(population)))

    uobj.evlolve = evolve
    return uobj


basic = (NSGA2, SPEA2, DeapEaSimple)
uniques = []

current_module = sys.modules[__name__]
for alg in basic:
    ualg = make_unique_version(alg)
    setattr(current_module, ualg.__name__, ualg)  # we need to create it in this submodules scope as well
    uniques.append(ualg)

all_algorithms = tuple(itertools.chain.from_iterable([basic, uniques]))
__all__ = [obj.__name__ for obj in all_algorithms]
