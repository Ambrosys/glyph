# Copyright: 2017, Markus Abel, Julien Gout, Markus Quade
# Licence: LGPL

import concurrent.futures
import copy
import enum
import itertools
import json
import logging
import os
import random
from functools import partial
from queue import Queue
from threading import Thread
from time import sleep

logging.captureWarnings(True)

import yaml

import deap.gp
import deap.tools
from deprecated import deprecated
import glyph.application
import glyph.gp.individual
import glyph.utils
import sympy
import zmq
from cache import DBCache
from scipy.optimize._minimize import _minimize_neldermead as nelder_mead


from glyph.assessment import const_opt
from glyph.cli._parser import *  # noqa
from glyph.gp.constraints import NullSpace, apply_constraints, build_constraints
from glyph.gp.individual import _constant_normal_form, add_sc, pretty_print, sc_mmqout, simplify_this
from glyph.observer import ProgressObserver
from glyph.utils import partition, key_set
from glyph.utils.argparse import *  # noqa
from glyph.utils.break_condition import break_condition
from glyph.utils.logging import print_params, load_config

logger = logging.getLogger(__name__)


class ExperimentProtocol(enum.EnumMeta):
    """Communication Protocol with remote experiments."""

    EXPERIMENT = "EXPERIMENT"
    SHUTDOWN = "SHUTDOWN"
    METADATA = "METADATA"
    CONFIG = "CONFIG"


class Communicator:
    def __init__(self, ip, port):
        """Holds the socket for 0mq communication.

        Args:
            ip: ip of the client
            port: port of the client
        """
        self._socket = zmq.Context.instance().socket(zmq.REQ)
        self.ip = ip
        self.port = port

    def connect(self):
        address = f"tcp://{self.ip}:{self.port}"
        logger.debug(f"Connecting to experiment on {address}")
        self._socket.connect(address)

    def send(self, msg, serializer=json):
        logger.log(logging.NOTSET, msg)
        self._socket.send(serializer.dumps(msg).encode("ascii"))

    def recv(self, serializer=json):
        msg = serializer.loads(self._socket.recv().decode("ascii"))
        logger.log(logging.NOTSET, msg)
        return msg


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
            self.assessment_runner.com.send(dict(action=ExperimentProtocol.SHUTDOWN))
            zmq.Context.instance().destroy()

    @classmethod
    def from_checkpoint(cls, file_name, com):
        """Create application from checkpoint file."""
        cp = glyph.application.load(file_name)
        args = cp["args"]
        gp_runner = cp["runner"]
        gp_runner.assessment_runner = RemoteAssessmentRunner(
            com,
            consider_complexity=args.consider_complexity,
            method=args.const_opt_method,
            options=args.options,
            caching=args.caching,
            simplify=args.simplify,
            persistent_caching=args.persistent_caching,
            chunk_size=args.chunk_size,
            multi_objective=args.multi_objective,
            send_symbolic=args.send_symbolic,
            reevaluate=args.re_evaluate,
        )
        app = cls(args, gp_runner, file_name, cp["callbacks"])
        app.pareto_fronts = cp["pareto_fronts"]
        app._initialized = True
        pset = build_pset_gp(args.primitives, args.structural_constants, args.sc_min, args.sc_max)
        Individual.pset = pset
        random.setstate(cp["random_state"])
        return app

    def checkpoint(self):
        """Checkpoint current state of evolution."""

        runner = copy.deepcopy(self.gp_runner)
        del runner.assessment_runner
        glyph.application.safe(
            self.checkpoint_file,
            args=self.args,
            runner=runner,
            random_state=random.getstate(),
            pareto_fronts=self.pareto_fronts,
            callbacks=self.callbacks,
        )
        logger.debug("Saved checkpoint to {}".format(self.checkpoint_file))


def handle_const_opt_config(args):
    smart_options = {
        "use": args.smart,
        "kw": {
            "threshold": args.smart_threshold,
            "step_size": args.smart_step_size,
            "min_stat": args.smart_min_stat,
        },
    }
    options = {"maxfev": args.max_fev_const_opt, "smart_options": smart_options}
    if args.const_opt_method == "hill_climb":
        options["directions"] = args.directions
        options["precision"] = args.precision
        options["target"] = args.target
    else:
        options["xatol"] = 10.0 ** (-args.precision)
        options["fatol"] = args.target
    args.options = options
    return args


def const_opt_options_transform(options):
    leastsq_options = dict()
    leastsq_options["xtol"] = options["xatol"]
    leastsq_options["ftol"] = options["fatol"]
    leastsq_options["max_nfev"] = options["maxfev"]
    return leastsq_options


def update_namespace(ns, up):
    """Update the argparse.Namespace ns with a dictionairy up."""
    return argparse.Namespace(**{**vars(ns), **up})


def handle_gpconfig(config, com):
    """Will try to load config from file or from remote and update the cli/default config accordingly."""
    if config.cfile:
        with open(config.cfile, "r") as cf:
            gpconfig = yaml.load(cf)
    elif config.remote:
        com.send(dict(action=ExperimentProtocol.CONFIG))
        gpconfig = com.recv()
    else:
        gpconfig = {}
    return update_namespace(config, gpconfig)


def build_pset_gp(primitives, structural_constants=False, cmin=-1, cmax=1):
    """Build a primitive set used in remote evaluation.

    Locally, all primitives correspond to the id() function.
    """
    pset = deap.gp.PrimitiveSet("main", arity=0)
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


class EvalQueue(Queue):
    def __init__(self, com, result_queue, expect):
        self.com = com
        self.result_queue = result_queue
        self.expect = expect

        super().__init__()

    def run(self, chunk_size=100):
        payloads = []
        keys = []

        def process(keys, payload_meta):
            payload, meta = zip(*payload_meta)
            if any(meta):
                self.com.send(dict(action=ExperimentProtocol.EXPERIMENT, payload=payload, meta=meta))
            else:
                self.com.send(dict(action=ExperimentProtocol.EXPERIMENT, payload=payload))
            fitnesses = self.com.recv()["fitness"]
            for key, fit in zip(keys, fitnesses):
                logger.debug("Writing result for key: {}".format(key))
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
                    logger.debug("Queueing key: {}".format(key))
                    payloads.append(payload_meta)
                    keys.append(key)
            if len(payloads) == min(self.expect, chunk_size):
                process(keys, payloads)
                payloads = []
                keys = []
        if payloads:
            process(keys, payloads)


def _no_const_opt(func, ind):
    return None, func(ind)


class RemoteAssessmentRunner:
    def __init__(
        self,
        com,
        consider_complexity=True,
        multi_objective=False,
        method="Nelder-Mead",
        options={"smart_options": {"use": False}},
        caching=True,
        persistent_caching=None,
        simplify=False,
        chunk_size=30,
        send_symbolic=False,
        reevaluate=False,
    ):
        """Contains assessment logic. Uses zmq connection to request evaluation."""
        self.com = com
        self.consider_complexity = consider_complexity
        self.multi_objective = multi_objective
        self.caching = caching
        self.cache = {} if persistent_caching is None else DBCache("glyph-remote", persistent_caching)
        self.make_str = (lambda i: str(simplify_this(i))) if simplify else str
        self.result_queue = {}
        self.send_symbolic = send_symbolic
        self.reevaluate = reevaluate
        self.evaluations = 0
        self.chunk_size = chunk_size
        if chunk_size > 30:
            logger.warning("Chunk size may cause performance issues.")

        if not self.send_symbolic:
            self.options = options
            self.method = {"hill_climb": glyph.utils.numeric.hill_climb}.get(method, nelder_mead)

            self.smart_options = options.get("smart_options")
            if self.smart_options["use"]:
                self.method = glyph.utils.numeric.SmartConstantOptimizer(
                    glyph.utils.numeric.hill_climb, **self.smart_options["kw"]
                )

            if self.multi_objective:
                self.const_optimizer = partial(
                    const_opt, lsq=True, **const_opt_options_transform(self.options)
                )
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
            key = sum(map(hash, payload))  # constants may have been simplified, not in payload anymore.
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
            error = (error,)

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

        if n > 0:  # main work is done here
            n_workers = min(n, self.chunk_size)

            # start queue and the broker
            self.queue = EvalQueue(self.com, self.result_queue, n)
            thread = Thread(target=self.queue.run, args=(n_workers,))
            thread.start()

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as exe:
                for k, future in zip(
                    calculate_duplicate_free,
                    exe.map(partial(self.measure, meta=meta), calculate_duplicate_free),
                ):
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

    @property
    @deprecated(reason="Use RemoteAssessmentRunner.com.send instead.", version="0.3.7")
    def send(self):
        """Backwards compatibility"""
        return self.com.send

    @property
    @deprecated(reason="Use RemoteAssessmentRunner.com.recv instead.", version="0.3.7")
    def recv(self):
        """Backwards compatibility"""
        return self.com.recv


class Individual(glyph.gp.individual.AExpressionTree):
    pass


class NDTree(glyph.gp.individual.ANDimTree):
    base = Individual

    def __hash__(self):
        return hash(hash(x) for x in self)


def make_callback(factories, args):
    return tuple(factory(args) for factory in factories)


def make_remote_app(callbacks=(), callback_factories=(), parser=None):
    parser = parser or get_parser()
    args, _ = parser.parse_known_args()
    if isinstance(parser, Parser):
        if hasattr(args, "gui") and args.gui:
            if GUI_AVAILABLE:
                parser = get_parser(get_gooey())
            else:
                raise ValueError(GUI_UNAVAILABLE_MSG)
    args = parser.parse_args()
    com = Communicator(args.ip, args.port)
    com.connect()
    workdir = os.path.dirname(os.path.abspath(args.checkpoint_file))
    if not os.path.exists(workdir):
        raise RuntimeError('Path does not exist: "{}"'.format(workdir))

    log_level = glyph.utils.logging.log_level(args.verbosity)
    glyph.utils.logging.load_config(
        config_file=args.logging_config, level=log_level, placeholders=dict(workdir=workdir)
    )
    if args.resume_file is not None:
        logger.debug("Loading checkpoint {}".format(args.resume_file))
        app = RemoteApp.from_checkpoint(args.resume_file, com)
    else:
        args = handle_const_opt_config(handle_gpconfig(args, com))
        try:
            pset = build_pset_gp(args.primitives, args.structural_constants, args.sc_min, args.sc_max)
        except AttributeError:
            raise AttributeError("You need to specify the pset")
        Individual.pset = pset
        mate = glyph.application.MateFactory.create(args, Individual)
        mutate = glyph.application.MutateFactory.create(args, Individual)
        select = glyph.application.SelectFactory.create(args)
        create_method = glyph.application.CreateFactory.create(args, Individual)

        ns = NullSpace(
            zero=args.constraints_zero, constant=args.constraints_constant, infty=args.constraints_infty
        )
        mate, mutate, Individual.create = apply_constraints(
            [mate, mutate, Individual.create], constraints=build_constraints(ns)
        )

        ndmate = partial(glyph.gp.breeding.nd_crossover, cx1d=mate)
        ndmutate = partial(glyph.gp.breeding.nd_mutation, mut1d=mutate)
        ndcreate = lambda size: [NDTree(create_method(args.ndim)) for _ in range(size)]
        NDTree.create_population = ndcreate
        algorithm_factory = partial(
            glyph.application.AlgorithmFactory.create, args, ndmate, ndmutate, select, ndcreate
        )
        assessment_runner = RemoteAssessmentRunner(
            com,
            method=args.const_opt_method,
            options=args.options,
            consider_complexity=args.consider_complexity,
            caching=args.caching,
            persistent_caching=args.persistent_caching,
            simplify=args.simplify,
            chunk_size=args.chunk_size,
            multi_objective=args.multi_objective,
            send_symbolic=args.send_symbolic,
            reevaluate=args.re_evaluate,
        )
        gp_runner = glyph.application.GPRunner(NDTree, algorithm_factory, assessment_runner)

        callbacks = glyph.application.DEFAULT_CALLBACKS + callbacks + make_callback(callback_factories, args)

        if args.send_meta_data:
            callbacks += (send_meta_data,)

        if args.animate:
            callbacks += (ProgressObserver(),)

        app = RemoteApp(args, gp_runner, args.checkpoint_file, callbacks=callbacks)

    bc = break_condition(ttl=args.ttl, target=args.target, max_iter=args.max_iter_total, error_index=0)
    logger.debug("Parameters:")
    print_params(logger.debug, vars(args))
    return app, bc, args


def send_meta_data(app):
    com = app.gp_runner.assessment_runner.com

    metadata = dict(generation=app.gp_runner.step_count)
    com.send(dict(action=ExperimentProtocol.METADATA, payload=metadata))
    logger.debug(com.recv())


def main():
    app, bc, args = make_remote_app()
    logger.info("Glyph-remote")
    app.run(break_condition=bc)


if __name__ == "__main__":
    main()
