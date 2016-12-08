from functools import partial, partialmethod

from deap.tools import selNSGA2, selBest
import numpy as np
import toolz

from glyph.gp.individual import AExpressionTree, ANDimTree, numpy_primitive_set
from glyph.gp.breeding import cxonepoint, nd_crossover, mutuniform, nd_mutation
from glyph.gp.algorithms import AgeFitness
from glyph.utils.numeric import rmse


pset = numpy_primitive_set(1, categories=('algebraic', 'symc'))
MyTree = type("MyTree", (AExpressionTree,), dict(pset=pset))
MyNDTree = type("MyNDTree", (ANDimTree,), dict(base=MyTree))
MyNDTree.create_population = partialmethod(MyNDTree.create_population, ndim=2)

mutate = partial(nd_mutation, mut1d=mutuniform(pset=pset))
mate = partial(nd_crossover, cx1d=cxonepoint())

algorithm = AgeFitness(mate, mutate, selNSGA2, MyNDTree.create_population)


def target(x):
    return np.array([f(x) for f in [lambda x: x**2, lambda x: x]])

x = np.linspace(-1, 1, 30)
y = target(x)


def evaluate_(individual, x, y):
    func = individual.compile()
    yhat = func(x)
    for i in range(len(yhat)):
        if np.isscalar(yhat[i]):
            yhat[i] = np.ones_like(y[i]) * yhat[i]
    yhat = np.array(yhat)
    return rmse(yhat, y),

evaluate = partial(evaluate_, x=x, y=y)


def update_fitness(population):
    invalid = [p for p in population if not p.fitness.valid]
    fitnesses = map(evaluate, invalid)
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit
    return population


def main():
    pop_size = 100
    loop = toolz.iterate(toolz.compose(algorithm.evolve, update_fitness), MyNDTree.create_population(pop_size))
    populations = list(toolz.take(10, loop))
    best = selBest(populations[-1], 1)[0]
    print(best)


if __name__ == "__main__":
    main()
