import os
import json
import logging
import random
import argparse
import copy
from functools import partial

import zmq
import deap.tools
import deap.gp

from glyph.gp import AExpressionTree
from glyph.utils import memoize
import glyph.application


class RemoteApp(glyph.application.Application):
    def run(self, break_condition=None):
        try:
            super().run(break_condition=break_condition)
        except KeyboardInterrupt:
            self.checkpoint()
        finally:
            self.assessment_runner.send(dict(action="SHUTDOWN"))
            zmq.Context.instance().destroy()

    @classmethod
    def from_checkpoint(cls, file_name, send, recv):
        """Create application from checkpoint file."""
        cp = glyph.application.load(file_name)
        gp_runner = cp['runner']
        gp_runner.assessment_runner = RemoteAssessmentRunner(send, recv)
        app = cls(cp['args'], cp['runner'], file_name)
        app.pareto_fronts = cp['pareto_fronts']
        app._initialized = True
        random.setstate(cp['random_state'])
        return app

    def checkpoint(self):
        """Checkpoint current state of evolution."""

        runner = copy.deepcopy(self.gp_runner)
        del runner.assessment_runner
        glyph.application.safe(self.checkpoint_file, args=self.args, runner=runner,
             random_state=random.getstate(), pareto_fronts=self.pareto_fronts)
        self.logger.debug('Saved checkpoint to {}'.format(self.checkpoint_file))


class Nestedspace(argparse.Namespace):
    def __setattr__(self, name, value):
        if '.' in name:
            group, name = name.split('.', 1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
        if '.' in name:
            group, name = name.split('.', 1)
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

    RemoteApp.add_options(parser)
    cp_group = parser.add_mutually_exclusive_group(required=False)
    cp_group.add_argument('--ndim', type=int, default=1)
    cp_group.add_argument('--resume', dest='resume_file', metavar='FILE', type=str,
                          help='continue previous run from a checkpoint file')
    cp_group.add_argument('-o', dest='checkpoint_file', metavar='FILE', type=str,
                          default=os.path.join('.', 'checkpoint.pickle'),
                          help='checkpoint to FILE (default: ./checkpoint.pickle)')
    parser.add_argument('--verbose', '-v', dest='verbosity', action='count', default=0,
                        help='set verbose output; raise verbosity level with -vv, -vvv, -vvvv')
    parser.add_argument('--logging', '-l', dest='logging_config', type=str, default='logging.yaml',
                        help='set config file for logging; overides --verbose (default: logging.yaml)')
    glyph.application.AlgorithmFactory.add_options(parser.add_argument_group('algorithm'))
    group_breeding = parser.add_argument_group('breeding')
    glyph.application.MateFactory.add_options(group_breeding)
    glyph.application.MutateFactory.add_options(group_breeding)
    glyph.application.SelectFactory.add_options(group_breeding)
    glyph.application.CreateFactory.add_options(group_breeding)

    return parser


def _send(socket, msg, serializer=json):
    socket.send(serializer.dumps(msg).encode('ascii'))


def _recv(socket, serializer=json):
    return serializer.loads(socket.recv().decode('ascii'))


def connect(ip, port):
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.connect('tcp://{ip}:{port}'.format(ip=ip, port=port))
    send = partial(_send, socket)
    recv = partial(_recv, socket)
    return send, recv


def handle_gpconfig(config, send, recv):
    if config.cfile:
        return None
    elif config.cli:
        return None
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


class RemoteAssessmentRunner:
    def __init__(self, send, recv, consider_complexity=True):
        super().__init__()
        self.send = send
        self.recv = recv
        self.consider_complexity = consider_complexity

    @memoize
    def measure(self, individual):
        self.send(dict(action="EXPERIMENT", payload=[str(t) for t in individual]))
        error = self.recv()["fitness"]
        if self.consider_complexity:
            fitness = *error, sum(map(len, individual))
        else:
            fitness = error
        return fitness

    def update_fitness(self, population, map=map):
        invalid = [p for p in population if not p.fitness.valid]
        fitnesses = map(self.measure, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit
        return len(invalid)

    def __call__(self, population):
        return self.update_fitness(population)


class Individual(AExpressionTree):
    pass


class NDTree(glyph.gp.individual.ANDimTree):
    base = Individual


def make_remote_app():
    parser = get_parser()
    args = parser.parse_args(namespace=Nestedspace())

    send, recv = connect(args.ip, args.port)
    workdir = os.path.dirname(os.path.abspath(args.checkpoint_file))
    if not os.path.exists(workdir):
        raise RuntimeError('Path does not exist: "{}"'.format(workdir))
    log_level = glyph.utils.logging.log_level(args.verbosity)
    glyph.utils.logging.load_config(config_file=args.logging_config, default_level=log_level, placeholders=dict(workdir=workdir))
    logger = logging.getLogger(__name__)

    if args.resume_file is not None:
        logger.debug('Loading checkpoint {}'.format(args.resume_file))
        app = RemoteApp.from_checkpoint(args.resume_file, send, recv)
    else:
        gpsettings = handle_gpconfig(args.config, send, recv)
        pset = build_pset_gp(gpsettings["primitives"])
        Individual.pset = pset
        mate = glyph.application.MateFactory.create(args, Individual)
        mutate = glyph.application.MutateFactory.create(args, Individual)
        select = glyph.application.SelectFactory.create(args)
        create_method = glyph.application.CreateFactory.create(args, Individual)
        ndmate = partial(glyph.gp.breeding.nd_crossover, cx1d=mate)
        ndmutate = partial(glyph.gp.breeding.nd_mutation, mut1d=mutate)
        ndcreate = lambda size: [NDTree(create_method(args.ndim)) for _ in range(size)]
        NDTree.create_population = ndcreate
        algorithm_factory = partial(glyph.application.AlgorithmFactory.create, args, ndmate, ndmutate, select, ndcreate)
        assessment_runner = RemoteAssessmentRunner(send, recv)
        gp_runner = glyph.application.GPRunner(NDTree, algorithm_factory, assessment_runner)
        app = RemoteApp(args, gp_runner, args.checkpoint_file)

    return app, args


def main():

    app, args = make_remote_app()
    app.run()


if __name__ == "__main__":
    main()
