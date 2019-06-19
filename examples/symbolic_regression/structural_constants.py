"""
structural constants
====================
"""


from functools import partial

import deap.gp
import deap.tools
import numpy as np

from glyph import gp
from glyph.utils import Memoize
from glyph.utils.numeric import nrmse, silent_numpy

pset = gp.numpy_primitive_set(arity=1, categories=["algebraic", "trigonometric", "exponential"])
pset = gp.individual.add_sc(pset, gp.individual.sc_mmqout)
Individual = gp.Individual(pset=pset)


@Memoize
@silent_numpy
def measure(ind):
    g = lambda x: x ** 2 - 1.1
    points = np.linspace(-1, 1, 100, endpoint=True)
    y = g(points)
    f = gp.individual.numpy_phenotype(ind)
    try:
        yhat = f(points)
    except TypeError:
        yhat = np.infty
    if np.isscalar(yhat):
        yhat = np.ones_like(y) * yhat
    return nrmse(y, yhat), len(ind.resolve_sc())


def update_fitness(population, map=map):
    invalid = [p for p in population if not p.fitness.valid]
    fitnesses = map(measure, invalid)
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit
    return population


def main():
    pop_size = 400

    mate = deap.gp.cxOnePoint
    expr_mut = partial(deap.gp.genFull, min_=0, max_=2)
    mutate = partial(deap.gp.mutUniform, expr=expr_mut, pset=Individual.pset)

    algorithm = gp.algorithms.AgeFitness(mate, mutate, deap.tools.selNSGA2, Individual.create_population)

    pop = update_fitness(Individual.create_population(pop_size))

    for gen in range(50):
        pop = algorithm.evolve(pop)
        pop = update_fitness(pop)
        best = deap.tools.selBest(pop, 1)[0]
        print(gp.individual.simplify_this(best), best.fitness.values)

        if best.fitness.values[0] <= 1e-3:
            break


if __name__ == "__main__":
    main()
