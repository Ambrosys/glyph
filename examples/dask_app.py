from glyph import application
import sys
import os
sys.path.append(os.path.dirname(__file__))
from minimal_example import complete_measure, Individual, pop_size


def update_fitness(population):
    global client
    invalid = [p for p in population if not p.fitness.valid]
    fitness = [client.submit(complete_measure, ind) for ind in invalid]
    fitness = client.gather(fitness)
    for ind, fit in zip(invalid, fitness):
        ind.fitness.values = fit
    return len(invalid)


def main():
    """Mit modul application -- noch kürzer."""

    runner = application.default_gprunner(Individual, update_fitness, algorithm='nsga2', mating='cxonepoint', mutation='mutuniform')
    runner.init(pop_size=pop_size)
    for gen in range(100):
        runner.step()
        print(runner.logbook.stream)
    for individual in runner.halloffame:
        print(individual)

if __name__ == "__main__":
    from distributed import Client
    client = Client()
    client.upload_file(os.path.join(os.path.dirname(__file__), "control_problem.py"))
    client.upload_file(os.path.join(os.path.dirname(__file__), "minimal_example.py"))
    main()
