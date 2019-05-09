import warnings
from functools import partial

import deap.gp
import deap.tools
import numpy as np

from glyph import gp
from glyph.application import create_stats
from glyph.assessment import const_opt
from glyph.observer import ProgressObserver
from glyph.utils import Memoize
from glyph.utils.numeric import silent_numpy, nrmse


pset = gp.numpy_primitive_set(arity=1, categories=['algebraic', 'trigonometric', 'exponential', 'symc'])
Individual = gp.Individual(pset=pset)


@silent_numpy
def error(ind, *args):
    g = lambda x: x**2 - 1.1
    points = np.linspace(-1, 1, 100, endpoint=True)
    y = g(points)
    f = gp.individual.numpy_phenotype(ind)
    yhat = f(points, *args)

    if np.isscalar(yhat):
        yhat = np.ones_like(y) * yhat
    return nrmse(y, yhat)


@Memoize
def measure(ind):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        popt, err_opr = const_opt(error, ind)
    ind.popt = popt
    return err_opr, len(ind)


def update_fitness(population, map=map):
    invalid = [p for p in population if not p.fitness.valid]
    fitnesses = map(measure, invalid)
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit
    return population


class Mock():
    pass


def main():
    pop_size = 20

    observer = ProgressObserver()
    mate = deap.gp.cxOnePoint
    expr_mut = partial(deap.gp.genFull, min_=0, max_=2)
    mutate = partial(deap.gp.mutUniform, expr=expr_mut, pset=Individual.pset)

    algorithm = gp.algorithms.AgeFitness(mate, mutate, deap.tools.selNSGA2, Individual.create_population)
    stats = create_stats(2)
    logbook = deap.tools.Logbook()
    pop = update_fitness(Individual.create_population(pop_size))

    for gen in range(20):
        pop = algorithm.evolve(pop)
        pop = update_fitness(pop)

        records = stats.compile(pop)
        logbook.record(gen=gen, **records)
        front = deap.tools.ParetoFront()
        front.update(pop)

        app = Mock()
        app.gp_runner = Mock()
        app.gp_runner.pareto_front = front
        app.logbook = logbook
        observer(app)


if __name__ == "__main__":
    main()
