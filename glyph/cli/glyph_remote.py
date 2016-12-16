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

from glyph.application import Application

class RemoteApp(Application):
    def run(self, breakpoint=None):
        try:
            super().run(breakpoint=breakpoint)
        except KeyboardInterrupt:
            self.checkpoint()
        finally:
            send(dict(action="SHUTDOWN"))

    @property
    def send(self):
        return self.assessment_runner.send

    @property:
    def recv(self):
        return self.assessment_runner.recv



class Nestedspace(argparse.Namespace):
    def __setattr__(self, name, value):
        if '.' in name:
            group,name = name.split('.',1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
        if '.' in name:
            group,name = name.split('.',1)
            try:
                ns = self.__dict__[group]
            except KeyError:
                raise AttributeError
            return getattr(ns, name)
        else:
            raise AttributeError

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5555, help='Port for the zeromq communication (default: 5555)')
    parser.add_argument('--ip', type=str, default="localhost", help='IP of the client (default: localhost)')

    config = parser.add_argument_group('config')
    group = config.add_mutually_exclusive_group()
    group.add_argument('--remote', action='store_true', dest='config.remote', default=True, help='Request GP configs from experiment handler.')
    group.add_argument('--cli', action='store_true', dest='config.cli', default=False, help='Read GP configs from command line.')
    group.add_argument('--cfile', dest='config.cfile', type=argparse.FileType('r'), help='Read GP configs from file')

    return parser

def _send(socket, msg, serializer=json):
    socket.send(serializer.dumps(msg).encode('ascii'))


def _recv(socket, serializer=json):
    return serializer.loads(socket.recv().decode('ascii'))


def connect(ip, port):
    socket = zmq.Context().socket(zmq.REQ)
    socket.connect('tcp://{ip}:{port}'.format(ip=ip, port=port))
    send = partial(_send, socket)
    recv = partial(_recv, socket)
    return send, recv


def handle_gpconfig(config, send, recv):
    if config.cfile:
        pass
    elif config.cli:
        pass
    else:
        send(dict(action="CONFIG"))
        gpconfig = recv()
    return gpconfig


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

    parser = get_parser()
    args = parser.parse_args(namespace=Nestedspace())

    send, recv = connect(args.ip, args.port)

    try:
        gpsettings = handle_gpconfig(args.config, send, recv)
        print(gpsettings)
        exit()

        pset = build_pset_gp(gpsettings["primitives"])
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
