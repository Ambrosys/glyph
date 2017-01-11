import os
import json
import logging
import random
import argparse
import copy
from functools import partial

import zmq
import yaml
import deap.tools
import deap.gp
import toolz
import numpy as np

from glyph.gp import AExpressionTree
from glyph.utils import Memoize
from glyph.utils.logging import print_params
from glyph.utils.argparse import readable_file
import glyph.application


class RemoteApp(glyph.application.Application):
    def run(self, break_condition=None):
        """For details see application.Application.
        Will checkpoint and close zmq connection on keyboard interruption.
        """
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
        gp_runner.assessment_runner = RemoteAssessmentRunner(send, recv, max_steps=cp['args'].hill_steps, directions=cp['args'].directions,
                                         consider_complexity=cp['args'].consider_complexity, precision=cp['args'].precision, caching=cp['args'].caching)
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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5555, help='Port for the zeromq communication (default: 5555)')
    parser.add_argument('--ip', type=str, default="localhost", help='IP of the client (default: localhost)')

    config = parser.add_argument_group('config')
    group = config.add_mutually_exclusive_group()
    group.add_argument('--remote', action='store_true', dest='remote', default=False, help='Request GP configs from experiment handler.')
    group.add_argument('--cfile', dest='cfile', type=readable_file, help='Read GP configs from file')

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

    ass_group = parser.add_argument_group('assessment')
    ass_group.add_argument('--directions', type=int, default=5, help='Number of directions to try in stochastic hill climber (default: 5)')
    ass_group.add_argument('--hill_steps', type=int, default=5, help='Number of iterations of stochastic hill climber (default: 5)')
    ass_group.add_argument('--consider_complexity', type=bool, default=True, help='Consider the complexity of solutions for MOO (default: True)')
    ass_group.add_argument('--caching', type=bool, default=True, help='Cache evaluation (default: True)')
    ass_group.add_argument('--precision', type=int, default=3, help='Precision of constants (default: 3)')
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


def update_namespace(ns, up):
    """Update the argparse.Namespace ns with a dictionairy up.
    """
    return argparse.Namespace(**{**vars(ns), **up})


def handle_gpconfig(config, send, recv):
    """Will try to load config from file or from remote and update the cli/default config accordingly.
    """
    if config.cfile:
        with open(config.cfile, 'r') as cf:
            gpconfig = yaml.load(cf)
    elif config.remote:
        send(dict(action="CONFIG"))
        gpconfig = recv()
    else:
        gpconfig = {}
    return update_namespace(config, gpconfig)


def build_pset_gp(primitives):
    """Build a primitive set used in remote evaluation. Locally, all primitives correspond to the id() function.
    """
    pset = deap.gp.PrimitiveSet('main', arity=0)
    pset.constants = set()
    for fname, arity in primitives.items():
        if arity > 0:
            func = lambda *args: args
            pset.addPrimitive(func, arity, name=fname)
        elif arity == 0:
            pset.addTerminal(fname, name=fname)
            pset.arguments.append(fname)
        else:
            pset.addTerminal(fname, name=fname)
            pset.constants.add(fname)
    if len(pset.terminals) == 0:
        raise RuntimeError("Pset needs at least one terminal node. You may have forgotten to specify it.")
    return pset


class hashabledict(dict):
    """We can use this as dict key"""
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def default_constants(individual, default=1):
    """Finds all constants which are used in the individual and tries to inherit old values (from parents).
    If a constant cannot be inherited, the default value will be used as initial guess.
    """
    constants_in_ind = {k for k in individual.base.pset.constants if any(k in str(i) for i in individual)}
    old_values = getattr(individual, "constants", {})
    return hashabledict({k: old_values.get(k, default) for k in constants_in_ind})  # try hotstart = inherited values


class RemoteAssessmentRunner:
    def __init__(self, send, recv, consider_complexity=True, max_steps=5, directions=5, caching=True, precision=3):
        """Contains assessment logic. Uses zmq connection to request evaluation.
        Constant optimization is done using a stochastic hill climber.
        """
        self.send = send
        self.recv = recv
        self.consider_complexity = consider_complexity
        self.max_steps = max_steps
        self.directions = directions
        self.precision = precision
        if caching:
            self.evaluate = Memoize(self.evaluate)

    def evaluate(self, individual, constants=None):
        """Evaluate a single individual.
        """
        constants = constants or {}
        self.evaluations += 1
        payload = [str(t) for t in individual]
        for k, v in constants.items():
            payload = [s.replace(k, str(v)) for s in payload]
        self.send(dict(action="EXPERIMENT", payload=payload))
        error = self.recv()["fitness"]
        return error

    def hill_climb(self, individual, rng=np.random):
        """Stochastic hill climber for constant optimization.
        Try self.directions different solutions per iteration to select a new best individual.
        This iterates self.max_steps times.
        """
        constants = default_constants(individual)
        memory = {constants: self.evaluate(individual, constants)}
        if len(constants.keys()) == 0:
            return self.evaluate(individual, constants)

        for _ in range(self.max_steps):
            for _ in range(self.directions):
                c = toolz.valmap(lambda x: tweak(x, self.precision), constants, factory=hashabledict)
                error = self.evaluate(individual, c)
                memory[c] = error
            constants = min(memory, key=memory.get)  # argmin for dictionaries
            memory = {constants: memory[constants]}
        individual.constants = constants

        return memory[constants]

    def measure(self, individual):
        """Construct fitness for given individual.
        """
        error = self.hill_climb(individual)
        if self.consider_complexity:
            fitness = *error, sum(map(len, individual))
        else:
            fitness = error
        return fitness

    def update_fitness(self, population, map=map):
        self.evaluations = 0
        invalid = [p for p in population if not p.fitness.valid]
        fitnesses = map(self.measure, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit
        return self.evaluations

    def __call__(self, population):
        return self.update_fitness(population)


def tweak(x, p, rng=np.random):
    """ x = round(x + xi, p) with xi ~ N(0, sqrt(x)+10**(-p))
    """
    return round(x+rng.normal(scale=np.sqrt(abs(x))+10**(-p)), p)


class Individual(AExpressionTree):
    pass


class NDTree(glyph.gp.individual.ANDimTree):
    base = Individual


def make_remote_app():
    parser = get_parser()
    args = parser.parse_args()

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
        args = handle_gpconfig(args, send, recv)
        try:
            pset = build_pset_gp(args.primitives)
        except AttributeError:
            raise AttributeError("You need to specify the pset")
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
        assessment_runner = RemoteAssessmentRunner(send, recv, max_steps=args.hill_steps, directions=args.directions,
                                                   consider_complexity=args.consider_complexity, precision=args.precision, caching=args.caching)
        gp_runner = glyph.application.GPRunner(NDTree, algorithm_factory, assessment_runner)
        app = RemoteApp(args, gp_runner, args.checkpoint_file)

    return app, args


def main():

    app, args = make_remote_app()
    logger = logging.getLogger(__name__)
    print_params(logger.info, vars(args))
    app.run()

if __name__ == "__main__":
    main()
