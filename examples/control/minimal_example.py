"""minimal gp application."""

import numpy as np

from glyph import application
from glyph import assessment
from glyph import gp
from glyph import utils
from toolz import partial, compose

import sys
import os
sys.path.append(os.path.dirname(__file__))
import control_problem

pop_size = 10


# Define genotype data structure and phenotype function.
class Individual(gp.AExpressionTree):
    """The gp representation (genotype) of the actuator for the control problem."""

    pset = gp.sympy_primitive_set(categories=['algebraic', 'trigonometric', 'exponential'],
                                  arguments=['y0', 'y1'])


def phenotype(individual):
    """Produce phenotype from Individual."""
    return gp.sympy_phenotype(individual)

# Setup dynamic system.
x = np.linspace(0.0, 2.0 * np.pi, 2000, dtype=np.float64)
dynsys = partial(control_problem.anharmonic_oscillator, omega=1.0, c=3.0 / 8.0, k=0.0)
# Define target of control.
target = np.sin(x)
# Define measure.
trajectory = compose(partial(control_problem.integrate, yinit=[1.0, 0.0], x=x), dynsys, phenotype)
rmse = partial(utils.numeric.rmse, target)
dynsys_measure = assessment.measure(rmse, pre=compose(lambda arr: arr[0], trajectory))
complete_measure = assessment.measure(dynsys_measure, len, post=assessment.replace_nan)


def update_fitness(population):
    invalid = [p for p in population if not p.fitness.valid]
    fitnesses = map(complete_measure, invalid)
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit
    return len(invalid)


def main1():
    """Komplett ohne modul application."""
    import deap

    mate = deap.gp.cxOnePoint
    expr_mut = partial(deap.gp.genFull, min_=0, max_=2)
    mutate = partial(deap.gp.mutUniform, expr=expr_mut, pset=Individual.pset)
    algorithm = gp.NSGA2(mate, mutate)

    population = Individual.create_population(pop_size)
    update_fitness(population)
    for gen in range(10):
        population = algorithm.evolve(population)
        update_fitness(population)
        print('generation:', gen)
    print('Solutions:', population)


def main2():
    """Mit modul application (aber ohne deap)."""

    mate = application.MateFactory.create(dict(mating='cxonepoint', mating_max_height=20), Individual)
    mutate = application.MutateFactory.create(dict(mutation='mutuniform', mutation_max_height=20), Individual)
    select = application.SelectFactory.create(dict(select='nsga2'))
    create = application.CreateFactory.create(dict(create_method='halfandhalf', create_min_height=1, create_max_height=4), Individual)
    algorithm_config = dict(algorithm='nsga2', crossover_prob=0.5,  mutation_prob=0.2, tournament_size=2)
    algorithm_factory = partial(application.AlgorithmFactory.create, algorithm_config, mate, mutate, select, create)
    runner = application.GPRunner(Individual, algorithm_factory, update_fitness)

    runner.init(pop_size=pop_size)
    for gen in range(10):
        runner.step()
        print(runner.logbook.stream)
    for individual in runner.halloffame:
        print(individual)


def main3():
    """Mit modul application -- noch kürzer."""

    runner = application.default_gprunner(Individual, update_fitness, algorithm='nsga2', mating='cxonepoint', mutation='mutuniform')
    runner.init(pop_size=pop_size)
    for gen in range(10):
        runner.step()
        print(runner.logbook.stream)
    for individual in runner.halloffame:
        print(individual)


if __name__ == '__main__':
    print('\n=== main1 ===')
    main1()
    print('\n=== main2 ===')
    main2()
    print('\n=== main3 ===')
    main3()
