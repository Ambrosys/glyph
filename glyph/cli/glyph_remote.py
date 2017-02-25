# Copyright: 2017, Markus Abel, Julien Gout, Markus Quade
# Licence: LGPL

import os
import json
import logging
import random
import argparse
import copy
import itertools
from functools import partial, wraps
from threading import Thread
import concurrent.futures
from queue import Queue
from time import sleep

import zmq
import yaml
import deap.tools
import deap.gp
import toolz
import numpy as np

from glyph.gp import AExpressionTree
from glyph.utils.logging import print_params
from glyph.utils.argparse import readable_file
from glyph.utils.break_condition import BreakCondition
from glyph.assessment import tuple_wrap, const_opt_scalar
from glyph.gp.individual import simplify_this, add_sc, sc_mmqout
from glyph.gp.constraints import build_constraints, apply_constraints, NullSpace
import glyph.application
import glyph.utils


def partition(pred, iterable):
    """Use a predicate to partition entries into false entries and true entries"""
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = itertools.tee(iterable)
    return itertools.filterfalse(pred, t1), filter(pred, t2)


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
                                         consider_complexity=cp['args'].consider_complexity, precision=cp['args'].precision, caching=cp['args'].caching,
                                         send_all=cp['args'].send_all, simplify=cp['args'].simplify)
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
    ass_group.add_argument('--send_all', action='store_true', default=False, help='Send all invalid individuals at once. (default: False)')
    ass_group.add_argument('--simplify', type=bool, default=True, help='Simplify expression before sending them. (default: True)')
    ass_group.add_argument('--consider_complexity', type=bool, default=True, help='Consider the complexity of solutions for MOO (default: True)')
    ass_group.add_argument('--caching', type=bool, default=True, help='Cache evaluation (default: True)')
    ass_group.add_argument('--max_iter_const_opt', type=int, default=100, help='Maximum number of iterations for constant optimization (default: 100)')
    ass_group.add_argument('--directions', type=int, default=5, help='Directions for the stochastic hill-climber (default: 5 only used in conjunction with --const_opt_method hill_climb)')
    ass_group.add_argument('--precision', type=int, default=3, help='Precision of constants (default: 3)')
    ass_group.add_argument('--const_opt_method', choices=['hill_climb', 'Nelder-Mead'], default='Nelder-Mead', help='Algorithm to optimize constants given a structure (default: Nelder-Mead)')
    ass_group.add_argument('--structural_constants', action='store_true', default=False, help='Make use of structural constants. (default: False)')
    ass_group.add_argument('--sc_min', type=float, default=-1, help='Minimum value of sc for scaling. (default: -1)')
    ass_group.add_argument('--sc_max', type=float, default=1, help='Maximum value of sc for scaling. (default: 1)')


    break_condition = parser.add_argument_group('break condition')
    break_condition.add_argument('--ttl', type=int, default=-1, help='Time to life (in seconds) until soft shutdown. -1 = no ttl (default: -1)')
    break_condition.add_argument('--target', type=float, default=0, help='Target error used in stopping criteria (default: 0)')
    break_condition.add_argument('--max_iter_total', type=float, default=np.infty, help='Target error used in stopping criteria (default: np.infty)')

    constraints = parser.add_argument_group('constraints')
    constraints.add_argument('--constraints_zero', type=bool, default=True, help='Discard zero individuals (default: True)')
    constraints.add_argument('--constraints_constant', type=bool, default=True, help='Discard constant individuals (default: False)')
    constraints.add_argument('--constraints_infty', type=bool, default=True, help='Discard individuals with infinities (default: True)')
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


def handle_const_opt_config(args):
    options = {'maxiter': args.max_iter_const_opt}
    if args.const_opt_method == 'hill_climb':
        options['directions'] = args.directions
        options['precision'] = args.precision
        options['target'] = args.target
    else:
        options['xatol'] = 10**(-args.precision)
        options['fatol'] = args.target
    args.options = options
    return args


def update_namespace(ns, up):
    """Update the argparse.Namespace ns with a dictionairy up."""
    return argparse.Namespace(**{**vars(ns), **up})


def handle_gpconfig(config, send, recv):
    """Will try to load config from file or from remote and update the cli/default config accordingly."""
    if config.cfile:
        with open(config.cfile, 'r') as cf:
            gpconfig = yaml.load(cf)
    elif config.remote:
        send(dict(action="CONFIG"))
        gpconfig = recv()
    else:
        gpconfig = {}
    return update_namespace(config, gpconfig)


def build_pset_gp(primitives, structural_constants=False):
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

    if structural_constants:
        f = partial(sc_mmqout, cmin=args.sc_min, cmax=args.sc_max)
        pset = add_sc(pset, f)
    return pset


class MyQueue(Queue):
    def __init__(self, send, recv, result_queue, expect):
        self.recv = recv
        self.send = send
        self.result_queue = result_queue
        self.expect = expect
        self.logger = logging.getLogger(self.__class__.__name__)
        super().__init__()

    def run(self):
        payloads = []
        keys = []

        def process(keys, payloads):
            self.send(dict(action="EXPERIMENT_ALL", payload=payloads))
            fitnesses = self.recv()["fitness"]
            for key,fit in zip(keys, fitnesses):
                self.logger.debug("Writing result for key: {}".format(key))
                self.result_queue[key] = fit

        while self.expect > 0:
            key_payload = self.get()

            if key_payload is None:
                self.expect -= 1
            else:
                key,payload = key_payload
                payloads.append(payload)
                keys.append(key)
            if len(payloads) == self.expect:
                process(keys, payloads)
                payloads = []
                keys = []

class RemoteAssessmentRunner:
    def __init__(self, send, recv, consider_complexity=True, method='Nelder-Mead', options={}, caching=True, simplify=True, send_all=False):
        """Contains assessment logic. Uses zmq connection to request evaluation."""
        self.send = send
        self.recv = recv
        self.consider_complexity = consider_complexity
        self.options = options
        self.method = {'hill-climb': glyph.utils.numeric.hill_climb}.get(method, 'Nelder-Mead')
        self.caching = caching
        self.cache = {}
        self.send_all = send_all
        self.make_str = lambda i: str(simplify_this(i)) if simplify else str
        self.result_queue = {}
        #self.logger = logging.getLogger(self.__class__.__name__)

    def predicate(self, ind):
        """Does this individual need to be evaluated?"""
        return self.caching and self._hash(ind) in self.cache

    def _hash(self, ind):
        return json.dumps([self.make_str(t) for t in ind])

    def _evaluate_callback():
        while True:
            payload = self._queue.get()
            if payload == None:
                break

        data = self.recv()["fitness"]

    def evaluate_single(self, individual, *consts):
        """Evaluate a single individual."""
        payload = [self.make_str(t) for t in individual]
        for k, v in zip(individual.pset.constants, consts):
            payload = [s.replace(k, str(v)) for s in payload]

        key = sum(map(hash, payload))   # constants may have been simplified, not in payload anymore.
        self.queue.put((key, payload))
        self.evaluations += 1

        result = None
        while result is None:
            sleep(0.1)
            #self.logger.debug("Waiting for result for key: {}".format(key))
            result = self.result_queue.get(key)
        return result

    def measure(self, individual):
        """Construct fitness for given individual."""
        popt, error = const_opt_scalar(self.evaluate_single, individual, method=glyph.utils.numeric.hill_climb, options=self.options)
        self.queue.put(None)
        individual.popt = popt
        if self.consider_complexity:
            fitness = error, sum(map(len, individual))
        else:
            fitness = error,
        return fitness

    def evaluate_all(self, pop):
        payload = [[self.make_str(t) for t in ind] for ind in pop]
        self.send(dict(action="EXPERIMENT_ALL", payload=payload))
        errors = self.recv()["fitness"]
        self.evaluations += len(payload)
        fitnesses = zip(errors, sum(map(len, individual)))
        return list(fitnesses)

    def update_fitness(self, population, map=map):
        self.evaluations = 0

        invalid = [p for p in population if not p.fitness.valid]

        calculate, cached = map(list, partition(self.predicate, invalid))

        cached_fitness = [self.cache[self._hash(ind)] for ind in cached]
        calculate_fitness = []
        if len(calculate) > 0:
            self.queue = MyQueue(self.send, self.recv, self.result_queue, len(calculate))
            thread = Thread(target=self.queue.run)
            thread.start()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(calculate)) as executor:
                futures = {executor.submit(self.measure, ind): ind for ind in calculate}
                for future in futures:
                    calculate_fitness.append(future.result())
            thread.join()
            del self.queue

        # save to cache
        for key, fit in zip(map(self._hash, calculate), calculate_fitness):
            self.cache[key] = fit

        # assign fitness to individuals
        for ind, fit in zip(cached + calculate, cached_fitness + calculate_fitness):
            ind.fitness.values = fit

        return self.evaluations


    def __call__(self, population):
        return self.update_fitness(population)


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
        args = handle_const_opt_config(handle_gpconfig(args, send, recv))
        try:
            pset = build_pset_gp(args.primitives, args.structural_constants)
        except AttributeError:
            raise AttributeError("You need to specify the pset")
        Individual.pset = pset
        mate = glyph.application.MateFactory.create(args, Individual)
        mutate = glyph.application.MutateFactory.create(args, Individual)
        select = glyph.application.SelectFactory.create(args)
        create_method = glyph.application.CreateFactory.create(args, Individual)

        ns = NullSpace(zero=args.constraints_zero, constant=args.constraints_constant, infty=args.constraints_infty)
        mate, mutate, Individual.create = apply_constraints([mate, mutate, Individual.create], constraints=build_constraints(ns))

        ndmate = partial(glyph.gp.breeding.nd_crossover, cx1d=mate)
        ndmutate = partial(glyph.gp.breeding.nd_mutation, mut1d=mutate)
        ndcreate = lambda size: [NDTree(create_method(args.ndim)) for _ in range(size)]
        NDTree.create_population = ndcreate
        algorithm_factory = partial(glyph.application.AlgorithmFactory.create, args, ndmate, ndmutate, select, ndcreate)
        assessment_runner = RemoteAssessmentRunner(send, recv, options=args.options, consider_complexity=args.consider_complexity,
                                                   caching=args.caching, send_all=args.send_all, simplify=args.simplify)
        gp_runner = glyph.application.GPRunner(NDTree, algorithm_factory, assessment_runner)
        app = RemoteApp(args, gp_runner, args.checkpoint_file)

    return app, args


def main():
    app, args = make_remote_app()
    logger = logging.getLogger(__name__)
    print_params(logger.info, vars(args))
    break_condition = BreakCondition(ttl=args.ttl, target=args.target, max_iter=args.max_iter_total, error_index=0)
    app.run(break_condition=break_condition)

if __name__ == "__main__":
    main()
