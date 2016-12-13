import argparse
import zmq
import json
import numpy
import toolz
from functools import partial, wraps
import deap.tools
import deap.gp

from glyph.gp import numpy_primitive_set, AExpressionTree, algorithms
from glyph.utils import memoize


def _send(socket, msg, serializer=json):
    socket.send(serializer.dumps(msg).encode('ascii'))


def _recv(socket, serializer=json):
    return serializer.loads(socket.recv().decode('ascii'))


def handle_gp_config():
    pass


def build_pset_gp(primitives):
    """Build a primitive set used in remote evaluation. Locally, all primitives correspond to the id() function.
    """
    pset = deap.gp.PrimitiveSet('main', arity=0)
    for fname, arity in primitives.items():
        if arity > 0:
            func = lambda *args: args
            pset.addPrimitive(func, arity, name=fname)
        elif arity == 0:
            pset.addTerminal(fname, name=fname)
            pset.arguments.append(fname)
        else:
            raise ValueError("Wrong arity in primitive specification.")
    return pset


class AssessmentRunner:
    def __init__(self, send, recv, consider_complexity=True):
        super().__init__()
        self.send = send
        self.recv = recv
        self.consider_complexity = consider_complexity

    @memoize
    def measure(self, individual):
        self.send(dict(action="EXPERIMENT", payload=str(individual)))
        error = self.recv()["fitness"]
        if self.consider_complexity:
            fitness = *error, len(individual)
        else:
            fitness = error
        return fitness

    def update_fitness(self, population, map=map):
        invalid = [p for p in population if not p.fitness.valid]
        fitnesses = map(self.measure, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit
        return population


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5555, help='Port for the zeromq communication (default: 5555)')
    parser.add_argument('--ip', type=str, default="localhost", help='IP of the client (default: localhost)')

    gpconfig = parser.add_argument_group('gpconfig')
    group = gpconfig.add_mutually_exclusive_group()
    group.add_argument('--remote', action='store_true', default=True, help='Request GP configs from experiment handler.')
    group.add_argument('--cli', action='store_true', help='Read GP configs from command line.')
    group.add_argument('--cfile', type=argparse.FileType('r'), help='Read GP configs from file')

    args = parser.parse_args()

    socket = zmq.Context().socket(zmq.REQ)
    socket.connect('tcp://{ip}:{port}'.format(ip=args.ip, port=args.port))
    send = partial(_send, socket)
    recv = partial(_recv, socket)
    try:
        send(dict(action="CONFIG"))
        config = recv()
        print(config)

        pset = build_pset_gp(config["primitives"])

        Individual = type("Individual", (AExpressionTree,), dict(pset=pset))

        pop_size = config.get('pop_size', 100)
        gen = config.get('generations', 10)

        mate = deap.gp.cxOnePoint
        expr_mut = partial(deap.gp.genFull, min_=0, max_=2)
        mutate = partial(deap.gp.mutUniform, expr=expr_mut, pset=pset)

        algorithm = algorithms.AgeFitness(mate, mutate, deap.tools.selNSGA2, Individual.create_population)
        assessment_runner = AssessmentRunner(send, recv, config.get('consider_complexity', True))

        loop = toolz.iterate(toolz.compose(algorithm.evolve, assessment_runner.update_fitness), Individual.create_population(pop_size))
        populations = list(toolz.take(gen, loop))
        best = deap.tools.selBest(populations[-1], 1)[0]

        print(best)
    except KeyboardInterrupt:
        pass
    finally:
        send(dict(action="SHUTDOWN"))


if __name__ == "__main__":
    main()
