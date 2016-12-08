from glyph import gp
from glyph.utils.numeric import silent_numpy
from functools import partial
import numpy

import deap.tools
import toolz


class Individual(gp.AExpressionTree):
    """The gp representation (genotype) of the actuator for the control problem."""

    pset = gp.numpy_primitive_set(arity=1, categories=['algebraic', 'trigonometric', 'exponential'])


class Memoize:
    """Memoize(fn) - an instance which acts like fn but memoizes its arguments
       Will only work on functions with non-mutable arguments
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.fn(*args)
        return self.memo[args]


@Memoize
@silent_numpy
def meassure(ind):
    g = lambda x: x**2 - 1.1
    points = numpy.linspace(-1, 1, 100, endpoint=True)
    y = g(points)
    f = ind.compile()
    yhat = f(points)
    return numpy.sqrt(numpy.mean((yhat - y)**2)), len(ind)


def update_fitness(population, map=map):
    invalid = [p for p in population if not p.fitness.valid]
    fitnesses = map(meassure, invalid)
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit
    return population


def main():
    pop_size = 100

    mate = deap.gp.cxOnePoint
    expr_mut = partial(deap.gp.genFull, min_=0, max_=2)
    mutate = partial(deap.gp.mutUniform, expr=expr_mut, pset=Individual.pset)

    algorithm = gp.algorithms.AgeFitness(mate, mutate, deap.tools.selNSGA2, Individual.create_population)

    loop = toolz.iterate(toolz.compose(algorithm.evolve, update_fitness), Individual.create_population(pop_size))
    populations = list(toolz.take(10, loop))
    best = deap.tools.selBest(populations[-1], 1)[0]
    print(best)

if __name__ == "__main__":
    main()
