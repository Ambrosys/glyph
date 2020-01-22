"""
Motifs
======
"""

from functools import partial

import numpy as np
import deap.gp
import deap.tools

from glyph import gp
from glyph.assessment import const_opt
from glyph.utils import Memoize
from glyph.utils.numeric import silent_numpy, nrmse


pset = gp.numpy_primitive_set(arity=1, categories=["algebraic", "trigonometric", "exponential", "symc"])
Individual = gp.Individual(pset=pset)


class ADF(deap.gp.Primitive):
    def __init__(self, name, arity, variable_names=None):
        self.name = name
        self.arity = arity
        self.args = [deap.gp.__type__] * arity
        self.ret = deap.gp.__type__
        self.variable_names = variable_names or ["x_{}".format(i) for i in range(arity)]
        self._format()

    def _format(self):
        self.fmt = self.name
        for i, v in enumerate(self.variable_names):
            self.fmt = self.fmt.replace(v, "{{{0}}}".format(i))

    def format(self, *args):
        return self.fmt.format(*args)


def pprint_individual(ind):
    name = str(ind)
    for c in ind.const_opt:
        name = name.replace("Symc", str(c), 1)
    return name


@silent_numpy
def error(ind, *args):
    g = lambda x: x ** 2 - 1.1
    points = np.linspace(-1, 1, 100, endpoint=True)
    y = g(points)
    f = gp.individual.numpy_phenotype(ind)
    yhat = f(points, *args)

    if np.isscalar(yhat):
        yhat = np.ones_like(y) * yhat
    return nrmse(y, yhat)


@Memoize
def measure(ind):
    popt, err_opr = const_opt(error, ind)
    ind.popt = popt
    return err_opr, len(ind)


def update_fitness(population, map=map):
    invalid = [p for p in population if not p.fitness.valid]
    fitnesses = map(measure, invalid)
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit
    return population


MOTIFS = set()


def add_motif(ind, pset):
    name = repr(ind)
    if name not in MOTIFS:
        motif = ADF(pprint_individual(ind), len(pset.arguments))
        pset._add(motif)
        pset.context[motif.name] = motif
        pset.prims_count += 1
        MOTIFS.add(name)

    return pset


def main():
    pop_size = 20

    mate = deap.gp.cxOnePoint
    expr_mut = partial(deap.gp.genFull, min_=0, max_=2)
    mutate = partial(deap.gp.mutUniform, expr=expr_mut, pset=Individual.pset)

    algorithm = gp.algorithms.AgeFitness(mate, mutate, deap.tools.selNSGA2, Individual.create_population)

    pop = update_fitness(Individual.create_population(pop_size))

    for gen in range(10):
        pop = algorithm.evolve(pop)
        pop = update_fitness(pop)
        best = deap.tools.selBest(pop, 1)[0]
        print(gp.individual.simplify_this(best), best.fitness.values)
        Individual.pset = add_motif(best, Individual.pset)

        if best.fitness.values[0] <= 1e-3:
            break

    print(MOTIFS)


if __name__ == "__main__":
    main()
