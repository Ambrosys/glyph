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
import sympy
from scipy.optimize._minimize import _minimize_neldermead as nelder_mead
from cache import DBCache

from glyph.gp import AExpressionTree
from glyph.utils.logging import print_params
from glyph.utils.argparse import readable_file
from glyph.utils.break_condition import break_condition
from glyph.assessment import const_opt
from glyph.gp.individual import simplify_this, add_sc, sc_mmqout, pretty_print, _constant_normal_form
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
        args = cp['args']
        gp_runner = cp['runner']
        gp_runner.assessment_runner = RemoteAssessmentRunner(send, recv, consider_complexity=args.consider_complexity,
                                                            method=args.const_opt_method, options=args.options,
                                                            caching=args.caching, simplify=args.simplify,
                                                            persistent_caching=args.persistent_caching, chunk_size=args.chunk_size,
                                                            multi_objective=args.multi_objective, send_symbolic=args.send_symbolic,
                                                            reevaluate=args.re_evaluate)
        app = cls(args, gp_runner, file_name, cp['callbacks'])
        app.pareto_fronts = cp['pareto_fronts']
        app._initialized = True
        pset = build_pset_gp(args.primitives, args.structural_constants, args.sc_min, args.sc_max)
        Individual.pset = pset
        random.setstate(cp['random_state'])
        return app

    def checkpoint(self):
        """Checkpoint current state of evolution."""

        runner = copy.deepcopy(self.gp_runner)
        del runner.assessment_runner
        glyph.application.safe(self.checkpoint_file, args=self.args, runner=runner,
                               random_state=random.getstate(), pareto_fronts=self.pareto_fronts, callbacks=self.callbacks)
        self.logger.debug('Saved checkpoint to {}'.format(self.checkpoint_file))


def get_parser():
    parser = argparse.ArgumentParser(prog="glyph-remote")
    parser.add_argument('--port', type=int, default=5555, help='Port for the zeromq communication (default: 5555)')
    parser.add_argument('--ip', type=str, default="localhost", help='IP of the client (default: localhost)')
    parser.add_argument('--send_meta_data', action="store_true", default=False, help='Send metadata after each generation')
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
    ass_group.add_argument('--simplify', action="store_true", default=False, help='Simplify expression before sending them. (default: False)')
    ass_group.add_argument('--consider_complexity', type=bool, default=True, help='Consider the complexity of solutions for MOO (default: True)')
    ass_group.add_argument('--no_caching', dest="caching", action="store_false", default=True, help='Cache evaluation (default: False)')
    ass_group.add_argument('--persistent_caching', default=None, help='Key for persistent data base cache for caching between experiments (default: None)')
    ass_group.add_argument('--max_fev_const_opt', type=int, default=100, help='Maximum number of function evaluations for constant optimization (default: 100)')
    ass_group.add_argument('--directions', type=int, default=5, help='Directions for the stochastic hill-climber (default: 5 only used in conjunction with --const_opt_method hill_climb)')
    ass_group.add_argument('--precision', type=int, default=3, help='Precision of constants (default: 3)')
    ass_group.add_argument('--const_opt_method', choices=['hill_climb', 'Nelder-Mead'], default='Nelder-Mead', help='Algorithm to optimize constants given a structure (default: Nelder-Mead)')
    ass_group.add_argument('--structural_constants', action='store_true', default=False, help='Make use of structural constants. (default: False)')
    ass_group.add_argument('--sc_min', type=float, default=-1, help='Minimum value of sc for scaling. (default: -1)')
    ass_group.add_argument('--sc_max', type=float, default=1, help='Maximum value of sc for scaling. (default: 1)')
    ass_group.add_argument('--smart', action="store_true", default=False, help='Use smart constant optimization. (default: False)')
    ass_group.add_argument('--smart_step_size', type=int, default=10, help='Number of fev in iterative function optimization. (default: 10)')
    ass_group.add_argument('--smart_min_stat', type=int, default=10, help='Number of samples required prior to stopping (default: 10)')
    ass_group.add_argument('--smart_threshold', type=int, default=25, help='Quantile of improvement rate. Abort constant optimization if below (default: 25)')
    ass_group.add_argument('--chunk_size', type=int, default=30, help='Number of individuals send per single request. (default: 30)')
    ass_group.add_argument('--multi_objective', action="store_true", default=False, help='Returned fitness is multi-objective (default: False)')
    ass_group.add_argument('--send_symbolic', action="store_true", default=False, help='Send the expression with symbolic constants (default: False)')
    ass_group.add_argument('--re_evaluate', action="store_true", default=False, help='Re-evaluate old individuals (default: False)')

    break_condition = parser.add_argument_group('break condition')
    break_condition.add_argument('--ttl', type=int, default=-1, help='Time to life (in seconds) until soft shutdown. -1 = no ttl (default: -1)')
    break_condition.add_argument('--target', type=float, default=0, help='Target error used in stopping criteria (default: 0)')
    break_condition.add_argument('--max_iter_total', type=int, default=np.infty, help='Maximum number of function evaluations (default: np.infty)')

    constraints = parser.add_argument_group('constraints')
    constraints.add_argument('--constraints_zero', type=bool, default=True, help='Discard zero individuals (default: True)')
    constraints.add_argument('--constraints_constant', type=bool, default=True, help='Discard constant individuals (default: True)')
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
    smart_options = {"use": args.smart, "kw": {"threshold": args.smart_threshold, "step_size": args.smart_step_size, "min_stat": args.smart_min_stat}}
    options = {'maxfev': args.max_fev_const_opt, 'smart_options': smart_options}
    if args.const_opt_method == 'hill_climb':
        options['directions'] = args.directions
        options['precision'] = args.precision
        options['target'] = args.target
    else:
        options['xatol'] = 10.0**(-args.precision)
        options['fatol'] = args.target
    args.options = options
    return args


def const_opt_options_transform(options):
    leastsq_options = {}
    leastsq_options["xtol"] = options['xatol']
    leastsq_options["ftol"] = options['fatol']
    leastsq_options["max_nfev"] = options['maxfev']
    return leastsq_options


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


def build_pset_gp(primitives, structural_constants=False, cmin=-1, cmax=1):
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
        f = partial(sc_mmqout, cmin=cmin, cmax=cmax)
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

    def run(self, chunk_size=100):
        payloads = []
        keys = []

        def process(keys, payload_meta):
            payload, meta = zip(*payload_meta)
            if any(meta):
                self.send(dict(action="EXPERIMENT", payload=payload, meta=meta))
            else:
                self.send(dict(action="EXPERIMENT", payload=payload))
            fitnesses = self.recv()["fitness"]
            for key, fit in zip(keys, fitnesses):
                self.logger.debug("Writing result for key: {}".format(key))
                self.result_queue[key] = fit

        while self.expect > 0:
            key_payload_meta = self.get()

            if key_payload_meta is None:
                self.expect -= 1
                if self.expect == 0:
                    break
            else:
                key, payload_meta = key_payload_meta
                if key not in self.result_queue:
                    self.logger.debug("Queueing key: {}".format(key))
                    payloads.append(payload_meta)
                    keys.append(key)
            if len(payloads) == min(self.expect, chunk_size):
                process(keys, payloads)
                payloads = []
                keys = []
        if payloads:
            process(keys, payloads)


def key_set(itr, key=hash):
    keys = map(key, itr)
    s = {k: v for k, v in zip(keys, itr)}
    return list(s.values())


def _no_const_opt(func, ind):
    return None, func(ind)


class RemoteAssessmentRunner:
    def __init__(self, send, recv, consider_complexity=True, multi_objective=False, method='Nelder-Mead', options={'smart_options': {'use': False}},
                 caching=True, persistent_caching=None, simplify=False, chunk_size=30, send_symbolic=False, reevaluate=False):
        """Contains assessment logic. Uses zmq connection to request evaluation."""
        self.send = send
        self.recv = recv
        self.consider_complexity = consider_complexity
        self.multi_objective = multi_objective
        self.caching = caching
        self.cache = {} if persistent_caching is None else DBCache("glyph-remote", persistent_caching)
        self.make_str = (lambda i: str(simplify_this(i))) if simplify else str
        self.result_queue = {}
        self.send_symbolic = send_symbolic
        self.reevaluate = reevaluate
        self.evaluations = 0
        self.chunk_size = min(chunk_size, 30)

        if not self.send_symbolic:
            self.options = options
            self.method = {'hill_climb': glyph.utils.numeric.hill_climb}.get(method, nelder_mead)

            self.smart_options = options.get('smart_options')
            if self.smart_options["use"]:
                self.method = glyph.utils.numeric.SmartConstantOptimizer(glyph.utils.numeric.hill_climb, **self.smart_options["kw"])

            if self.multi_objective:
                self.const_optimizer = partial(const_opt, lsq=True, **const_opt_options_transform(self.options))
            else:
                self.const_optimizer = partial(const_opt, method=self.method, options=self.options)

        else:
            self.const_optimizer = _no_const_opt

    def predicate(self, ind):
        """Does this individual need to be evaluated?"""
        return self.caching and self._hash(ind) in self.cache

    def _hash(self, ind):
        return json.dumps([self.make_str(t) for t in ind])

    def evaluate_single(self, individual, *consts, meta=None):
        """Evaluate a single individual."""
        payload = [self.make_str(t) for t in individual]
        if not self.send_symbolic:
            payload = [pretty_print(s, individual.pset.constants, consts) for s in payload]
            key = sum(map(hash, payload))   # constants may have been simplified, not in payload anymore.
        else:
            variables = [sympy.Symbol(s) for s in Individual.pset.arguments]
            normal_form = [_constant_normal_form(sympy.sympify(p), variables=variables) for p in payload]
            key = sum(map(hash, normal_form))

        self.queue.put((key, (payload, meta)))
        self.evaluations += 1

        result = None
        while result is None:
            sleep(0.1)
            result = self.result_queue.get(key)
        return result

    def measure(self, individual, meta=None):
        """Construct fitness for given individual."""
        popt, error = self.const_optimizer(self.evaluate_single, individual, f_kwargs=dict(meta=meta))
        if not self.multi_objective:
            error = error,

        self.queue.put(None)
        individual.popt = popt
        if self.consider_complexity:
            fitness = *error, sum(map(len, individual))
        else:
            fitness = error
        return fitness

    def update_fitness(self, population, meta=None):
        self.evaluations = 0
        meta = meta or {}

        if self.reevaluate:
            for p in population:
                del p.fitness.values

        invalid = [p for p in population if not p.fitness.valid]
        calculate, cached = map(list, partition(self.predicate, invalid))
        cached_fitness = [self.cache[self._hash(ind)] for ind in cached]
        calculate_duplicate_free = key_set(calculate, key=self._hash)
        # if we have duplicates in the calculate list, dont calculate these more than once.
        dup_free_cache = {}
        n = len(calculate_duplicate_free)

        if n > 0:             # main work is done here
            n_workers = min(n, self.chunk_size)

            # start queue and the broker
            self.queue = MyQueue(self.send, self.recv, self.result_queue, n)
            thread = Thread(target=self.queue.run, args=(n_workers,))
            thread.start()

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as exe:
                for k, future in zip(calculate_duplicate_free, exe.map(partial(self.measure, meta=meta), calculate_duplicate_free)):
                    dup_free_cache[self._hash(k)] = future
            thread.join()
            del self.queue

            calculate_fitness = [dup_free_cache[self._hash(k)] for k in calculate]
        else:
            calculate_fitness = []

        if self.caching:
            # save to cache
            for key, fit in zip(map(self._hash, calculate), calculate_fitness):
                self.cache[key] = fit

        # assign fitness to individuals
        for ind, fit in zip(cached + calculate, cached_fitness + calculate_fitness):
            ind.fitness.values = fit

        if self.reevaluate or not self.caching:
            self.result_queue = {}

        return self.evaluations

    def __call__(self, population, meta=None):
        meta = meta or {}
        return self.update_fitness(population, meta=meta)


class Individual(AExpressionTree):
    pass


class NDTree(glyph.gp.individual.ANDimTree):
    base = Individual

    def __hash__(self):
        return hash(hash(x) for x in self)

def make_callback(factories, args):
    return tuple(factory(args) for factory in factories)

def make_remote_app(callbacks=(), callback_factories=(), parser=None):
    parser = parser or get_parser()
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
            pset = build_pset_gp(args.primitives, args.structural_constants, args.sc_min, args.sc_max)
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
        assessment_runner = RemoteAssessmentRunner(send, recv, method=args.const_opt_method, options=args.options,
                                                   consider_complexity=args.consider_complexity, caching=args.caching, persistent_caching=args.persistent_caching,
                                                   simplify=args.simplify, chunk_size=args.chunk_size, multi_objective=args.multi_objective, send_symbolic=args.send_symbolic,
                                                   reevaluate=args.re_evaluate)
        gp_runner = glyph.application.GPRunner(NDTree, algorithm_factory, assessment_runner)

        callbacks = glyph.application.DEFAULT_CALLBACKS + callbacks + make_callback(callback_factories, args)
        if args.send_meta_data:
            callbacks += send_meta_data,

        app = RemoteApp(args, gp_runner, args.checkpoint_file, callbacks=callbacks)

    bc = break_condition(ttl=args.ttl, target=args.target, max_iter=args.max_iter_total, error_index=0)
    print_params(logger.info, vars(args))
    return app, bc, args


def send_meta_data(app):
    send = app.gp_runner.assessment_runner.send
    recv = app.gp_runner.assessment_runner.recv

    metadata = dict(generation=app.gp_runner.step_count)
    send(dict(action="METADATA", payload=metadata))
    recv()


def main():
    app, bc, args = make_remote_app()
    app.run(break_condition=bc)

if __name__ == "__main__":
    main()
