# Copyright: 2017, Markus Abel, Julien Gout, Markus Quade
# Licence: LGPL

import argparse
import copy
import itertools
import json
import logging
import os
import random
import sys
from functools import partial
from threading import Thread
from time import sleep

import numpy as np

import concurrent.futures
import deap.gp
import deap.tools
import enum
import glyph.application
import glyph.gp.individual
import glyph.utils
import sympy
import yaml
import zmq
from cache import DBCache
from glyph.assessment import const_opt
from glyph.gp.constraints import (NullSpace, apply_constraints,
                                  build_constraints)
from glyph.gp.individual import (_constant_normal_form, add_sc, pretty_print,
                                 sc_mmqout, simplify_this)
from glyph.observer import ProgressObserver
from glyph.utils.argparse import positive_int, is_positive_int, \
                            non_negative_int, is_non_negative_int, \
                            unit_interval, is_unit_interval, \
                            readable_file, is_readable_file, \
                            readable_yaml_file, is_readable_yaml_file, \
                            np_infinity_int, is_np_infinity_int
from glyph.utils.break_condition import break_condition
from glyph.utils.logging import print_params
from glyph.gui.glyph_gooey import get_gooey
from queue import Queue
from scipy.optimize._minimize import _minimize_neldermead as nelder_mead

logger = logging.getLogger(__name__)


class ExperimentProtocol(enum.EnumMeta):
    """Communication Protocol with remote experiments."""

    EXPERIMENT = "EXPERIMENT"
    SHUTDOWN = "SHUTDOWN"
    METADATA = "METADATA"
    CONFIG = "CONFIG"


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
            self.assessment_runner.send(dict(action=ExperimentProtocol.SHUTDOWN))
            zmq.Context.instance().destroy()

    @classmethod
    def from_checkpoint(cls, file_name, send, recv):
        """Create application from checkpoint file."""
        cp = glyph.application.load(file_name)
        args = cp["args"]
        gp_runner = cp["runner"]
        gp_runner.assessment_runner = RemoteAssessmentRunner(
            send,
            recv,
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
        self.logger.debug("Saved checkpoint to {}".format(self.checkpoint_file))

class GooeyOptionsArg(enum.Enum):
    POSITIVE_INT = {
        "validator": {
            "callback": is_positive_int,
            "message": "This is not a positive integer.",
        }
    }
    NON_NEGATIVE_INT = {
        "validator": {
            "callback": is_non_negative_int,
            "message": "This is not a non negative integer.",
        }
    }
    READABLE_FILE = {
        "validator": {
            "callback": is_readable_file,
            "message": "This is not a readable file.",
        }
    }

def get_parser(parser=None, gui=False):
    parser = parser or argparse.ArgumentParser(prog="glyph-remote")
    to_add_list = []
    parameter_list = []

    # Argparse.add_argument needs some flags followed by kwargs
    # - For each argument a kwargs dict is created
    # - This dict is extended for the gui if needed
    # - Than a tuple is constructed of the needed flags (as a list) and the dict.
    # - This list contains now all args argaprse.add_argument()/Gooey.add_argument needs
    # The list is added to a list containing the args for all argparse.arguments of a specifique group (tab)
    # and the reference to the group is stored as well (parameter_list and to_add_list).
    # In the end bot of these meta-groups are filled.
    main_list = []
    port_dict = dict(
        type=positive_int,
        default=5555,
        help="Port for the zeromq communication (default: 5555)",
    )
    if gui:
        port_dict.update(
            dict(
                gooey_options={
                    "validator": {
                        "callback": is_positive_int,
                        "message": "This should be a positive port number in the range of 0 - 65535.",
                    }
                }
            )
        )
    main_list.append((["--port"], port_dict))

    ip_dict = dict(
        type=str, default="localhost", help="IP of the client (default: localhost)"
    )
    main_list.append((["--ip"], ip_dict))

    send_meta_data_dict = dict(
        action="store_true", default=False, help="Send metadata after each generation"
    )
    main_list.append((["--send_meta_data"], send_meta_data_dict))

    gui_output_dict = dict(
        action="store_true",
        default=False,
        help="Additional gui output (default: False)",
    )
    main_list.append((["--gui-output"], gui_output_dict))

    verbose_dict = dict(
        dest="verbosity",
        choices=["", "v", "vv", "vvv", "vvvv"],
        default="v",
        help="set verbose output; raise verbosity level with -vv, -vvv, -vvvv from lv 1-3",
    )
    main_list.append((["--verbose", "-v"], verbose_dict))

    logging_dict = dict(
        dest="logging_config",
        type=str,
        default="logging.yaml",
        help="set config file for logging; overides --verbose (default: logging.yaml)",
    )
    if gui:
        logging_dict.update(dict(widget="FileChooser"))
    main_list.append((["--logging", "-l"], logging_dict))

    parameter_list.append(main_list)
    to_add_list.append(parser)

    group_list = []
    config = parser.add_argument_group("config")
    group = config.add_mutually_exclusive_group(required=True if gui else False)
    remote_dict = dict(
        action="store_true",
        dest="remote",
        default=False,
        help="Request GP configs from experiment handler.",
    )
    group_list.append((["--remote"], remote_dict))

    cfile_dict = dict(
        dest="cfile", type=readable_yaml_file, help="Read GP configs from file"
    )
    if gui:
        cfile_dict.update(
            dict(
                widget="FileChooser",
                gooey_options={
                    "validator": {
                        "callback": is_readable_yaml_file,
                        "message": "This should be a readable .yaml file.",
                    }
                },
            )
        )
    group_list.append((["--cfile"], cfile_dict))
    parameter_list.append(group_list)
    to_add_list.append(group)

    cp_group_list = []
    RemoteApp.add_options(parser)
    cp_group = parser.add_mutually_exclusive_group(required=True if gui else False)

    ndim_dict = dict(type=positive_int, default=1)
    if gui:
        ndim_dict.update(dict(gooey_options=GooeyOptionsArg.POSITIVE_INT.value))
    cp_group_list.append((["--ndim"], ndim_dict))

    resume_dict = dict(
        dest="resume_file",
        metavar="FILE",
        type=readable_file,
        help="continue previous run from a checkpoint file",
    )
    if gui:
        del resume_dict["metavar"]
        resume_dict.update(
            dict(widget="FileChooser", gooey_options=GooeyOptionsArg.READABLE_FILE.value)
        )
    cp_group_list.append((["--resume"], resume_dict))

    o_dict = dict(
        dest="checkpoint_file",
        metavar="FILE",
        type=str,
        default=os.path.join(".", "checkpoint.pickle"),
        help="checkpoint to FILE (default: ./checkpoint.pickle)",
    )
    if gui:
        del o_dict["metavar"]
        o_dict.update(dict(widget="FileChooser"))
    cp_group_list.append((["-o"], o_dict))

    parameter_list.append(cp_group_list)
    to_add_list.append(cp_group)

    glyph.application.AlgorithmFactory.add_options(
        parser.add_argument_group("algorithm")
    )
    group_breeding = parser.add_argument_group("breeding")
    glyph.application.MateFactory.add_options(group_breeding)
    glyph.application.MutateFactory.add_options(group_breeding)
    glyph.application.SelectFactory.add_options(group_breeding)
    glyph.application.CreateFactory.add_options(group_breeding)

    ass_group_list = []
    ass_group = parser.add_argument_group("assessment")
    simplify_dict = dict(
        action="store_true",
        default=False,
        help="Simplify expression before sending them. (default: False)",
    )
    ass_group_list.append((["--simplify"], simplify_dict))

    consider_complexity_dict = dict(
        action="store_false",
        default=True,
        help="Consider the complexity of solutions for MOO (default: True)",
    )
    ass_group_list.append((["--consider_complexity"], consider_complexity_dict))

    no_caching_dict = dict(
        dest="caching",
        action="store_false",
        default=True,
        help="Cache evaluation (default: False)",
    )
    ass_group_list.append((["--no_caching"], no_caching_dict))

    persistent_caching_dict = dict(
        default=None,
        help="Key for persistent data base cache for caching between experiments (default: None)",
    )
    ass_group_list.append((["--persistent_caching"], persistent_caching_dict))

    max_fev_const_opt_dict = dict(
        type=non_negative_int,
        default=100,
        help="Maximum number of function evaluations for constant optimization (default: 100)",
    )
    if gui:
        max_fev_const_opt_dict.update(dict(gooey_options=GooeyOptionsArg.NON_NEGATIVE_INT.value))
    ass_group_list.append((["--max_fev_const_opt"], max_fev_const_opt_dict))

    directions_dict = dict(
        type=positive_int,
        default=5,
        help="Directions for the stochastic hill-climber (default: 5 only used in conjunction with --const_opt_method hill_climb)",
    )
    if gui:
        max_fev_const_opt_dict.update(dict(gooey_options=GooeyOptionsArg.POSITIVE_INT.value))
    ass_group_list.append((["--directions"], directions_dict))

    precision_dict = dict(
        type=non_negative_int, default=3, help="Precision of constants (default: 3)"
    )
    if gui:
        precision_dict.update(dict(gooey_options=GooeyOptionsArg.NON_NEGATIVE_INT.value))
    ass_group_list.append((["--precision"], precision_dict))

    const_opt_method_dict = dict(
        choices=["hill_climb", "Nelder-Mead"],
        default="Nelder-Mead",
        help="Algorithm to optimize constants given a structure (default: Nelder-Mead)",
    )
    ass_group_list.append((["--const_opt_method"], const_opt_method_dict))

    structural_constants_dict = dict(
        action="store_true",
        default=False,
        help="Make use of structural constants. (default: False)",
    )
    ass_group_list.append((["--structural_constants"], structural_constants_dict))

    sc_min_dict = dict(
        type=float, default=-1, help="Minimum value of sc for scaling. (default: -1)"
    )
    ass_group_list.append((["--sc_min"], sc_min_dict))

    sc_max_dict = dict(
        type=float, default=1, help="Maximum value of sc for scaling. (default: 1)"
    )
    ass_group_list.append((["--sc_max"], sc_max_dict))

    smart_dict = dict(
        action="store_true",
        default=False,
        help="Use smart constant optimization. (default: False)",
    )
    ass_group_list.append((["--smart"], smart_dict))

    smart_step_size_dict = dict(
        type=non_negative_int,
        default=10,
        help="Number of fev in iterative function optimization. (default: 10)",
    )
    if gui:
        smart_step_size_dict.update(dict(gooey_options=GooeyOptionsArg.NON_NEGATIVE_INT.value))
    ass_group_list.append((["--smart_step_size"], smart_step_size_dict))

    smart_min_stat_dict = dict(
        type=non_negative_int,
        default=10,
        help="Number of samples required prior to stopping (default: 10)",
    )
    if gui:
        smart_min_stat_dict.update(dict(gooey_options=GooeyOptionsArg.NON_NEGATIVE_INT.value))
    ass_group_list.append((["--smart_min_stat"], smart_min_stat_dict))

    smart_threshold_dict = dict(
        type=non_negative_int,
        default=25,
        help="Quantile of improvement rate. Abort constant optimization if below (default: 25)",
    )
    if gui:
        smart_threshold_dict.update(dict(gooey_options=GooeyOptionsArg.NON_NEGATIVE_INT.value))
    ass_group_list.append((["--smart_threshold"], smart_threshold_dict))

    chunk_size_dict = dict(
        type=positive_int,
        default=30,
        help="Number of individuals send per single request. (default: 30)",
    )
    if gui:
        chunk_size_dict.update(dict(gooey_options=GooeyOptionsArg.POSITIVE_INT.value))
    ass_group_list.append((["--chunk_size"], chunk_size_dict))

    multi_objective_dict = dict(
        action="store_true",
        default=False,
        help="Returned fitness is multi-objective (default: False)",
    )
    ass_group_list.append((["--multi_objective"], multi_objective_dict))

    send_symbolic_dict = dict(
        action="store_true",
        default=False,
        help="Send the expression with symbolic constants (default: False)",
    )
    ass_group_list.append((["--send_symbolic"], send_symbolic_dict))

    re_evaluate_dict = dict(
        action="store_true",
        default=False,
        help="Re-evaluate old individuals (default: False)",
    )
    ass_group_list.append((["--re_evaluate"], re_evaluate_dict))

    parameter_list.append(ass_group_list)
    to_add_list.append(ass_group)

    break_condition_list = []
    break_condition = parser.add_argument_group("break condition")
    ttl_dict = dict(
        type=int,
        default=-1,
        help="Time to life (in seconds) until soft shutdown. -1 = no ttl (default: -1)",
    )
    break_condition_list.append((["--ttl"], ttl_dict))

    target_dict = dict(
        type=float,
        default=0,
        help="Target error used in stopping criteria (default: 0)",
    )
    break_condition_list.append((["--target"], target_dict))

    max_iter_total_dict = dict(
        type=np_infinity_int,
        default=np.infty,
        help="Maximum number of function evaluations (default: 'inf' [stands for np.infty])",
    )
    if gui:
        max_iter_total_dict.update(
            dict(
                gooey_options={
                    "validator": {
                        "callback": is_np_infinity_int,
                        "message": 'This is neither "inf" nor a natural number.',
                    }
                }
            )
        )
    break_condition_list.append((["--max_iter_total"], max_iter_total_dict))

    parameter_list.append(break_condition_list)
    to_add_list.append(break_condition)

    constraints_list = []
    constraints = parser.add_argument_group("constraints")
    constraints_zero_dict = dict(
        action="store_false",
        default=True,
        help="Discard zero individuals (default: True)",
    )
    constraints_list.append((["--constraints_zero"], constraints_zero_dict))

    constraints_constant_dict = dict(
        action="store_false",
        default=True,
        help="Discard constant individuals (default: True)",
    )
    constraints_list.append((["--constraints_constant"], constraints_constant_dict))

    constraints_infty_dict = dict(
        action="store_false",
        default=True,
        help="Discard individuals with infinities (default: True)",
    )
    constraints_list.append((["--constraints_infty"], constraints_infty_dict))
    parameter_list.append(constraints_list)
    to_add_list.append(constraints)

    observer_list = []
    observer = parser.add_argument_group("observer")
    animate_dict = dict(
        action="store_true",
        default=False,
        help="Animate the progress of evolutionary optimization. (default: False)",
    )
    observer_list.append((["--animate"], animate_dict))
    parameter_list.append(observer_list)
    to_add_list.append(observer)

    # All arguments are prepared, now we fill the parser
    for p_list, to_add in zip(parameter_list, to_add_list):
        # A pair consists of the arguments and the group to which they shall be added
        for flags, kwargs in p_list:
            to_add.add_argument(*flags, **kwargs)
            
    return parser


def _send(socket, msg, serializer=json):
    socket.send(serializer.dumps(msg).encode("ascii"))


def _recv(socket, serializer=json):
    return serializer.loads(socket.recv().decode("ascii"))


def connect(ip, port):
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.connect("tcp://{ip}:{port}".format(ip=ip, port=port))
    send = partial(_send, socket)
    recv = partial(_recv, socket)
    return send, recv


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
    leastsq_options = {}
    leastsq_options["xtol"] = options["xatol"]
    leastsq_options["ftol"] = options["fatol"]
    leastsq_options["max_nfev"] = options["maxfev"]
    return leastsq_options


def update_namespace(ns, up):
    """Update the argparse.Namespace ns with a dictionairy up."""
    return argparse.Namespace(**{**vars(ns), **up})


def handle_gpconfig(config, send, recv):
    """Will try to load config from file or from remote and update the cli/default config accordingly."""
    if config.cfile:
        with open(config.cfile, "r") as cf:
            gpconfig = yaml.load(cf)
    elif config.remote:
        send(dict(action=ExperimentProtocol.CONFIG))
        gpconfig = recv()
    else:
        gpconfig = {}
    return update_namespace(config, gpconfig)


def build_pset_gp(primitives, structural_constants=False, cmin=-1, cmax=1):
    """Build a primitive set used in remote evaluation. Locally, all primitives correspond to the id() function.
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
    def __init__(self, send, recv, result_queue, expect):
        self.recv = recv
        self.send = send
        self.result_queue = result_queue
        self.expect = expect

        super().__init__()

    def run(self, chunk_size=100):
        payloads = []
        keys = []

        def process(keys, payload_meta):
            payload, meta = zip(*payload_meta)
            if any(meta):
                self.send(dict(action=ExperimentProtocol.EXPERIMENT, payload=payload, meta=meta))
            else:
                self.send(dict(action=ExperimentProtocol.EXPERIMENT, payload=payload))
            fitnesses = self.recv()["fitness"]
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


def key_set(itr, key=hash):
    keys = map(key, itr)
    s = {k: v for k, v in zip(keys, itr)}
    return list(s.values())


def _no_const_opt(func, ind):
    return None, func(ind)


class RemoteAssessmentRunner:
    def __init__(
        self,
        send,
        recv,
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
            self.queue = EvalQueue(self.send, self.recv, self.result_queue, n)
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


class Individual(glyph.gp.individual.AExpressionTree):
    pass


class NDTree(glyph.gp.individual.ANDimTree):
    base = Individual

    def __hash__(self):
        return hash(hash(x) for x in self)


def make_callback(factories, args):
    return tuple(factory(args) for factory in factories)


def make_remote_app(callbacks=(), callback_factories=(), parser=None):
    if parser is None:
        if "--gui" in sys.argv:
            parser = get_parser(parser=get_gooey(RemoteApp), gui=True)
        else:
            parser = get_parser()
    args = parser.parse_args()
    send, recv = connect(args.ip, args.port)
    workdir = os.path.dirname(os.path.abspath(args.checkpoint_file))
    if not os.path.exists(workdir):
        raise RuntimeError('Path does not exist: "{}"'.format(workdir))
    args.__dict__["verbosity"] = len(args.verbosity)
    log_level = glyph.utils.logging.log_level(args.verbosity)
    glyph.utils.logging.load_config(
        config_file=args.logging_config, default_level=log_level, placeholders=dict(workdir=workdir)
    )
    logger = logging.getLogger(__name__)

    if args.resume_file is not None:
        logger.debug("Loading checkpoint {}".format(args.resume_file))
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
            send,
            recv,
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
    print_params(logger.info, vars(args))
    return app, bc, args


def send_meta_data(app):
    send = app.gp_runner.assessment_runner.send
    recv = app.gp_runner.assessment_runner.recv

    metadata = dict(generation=app.gp_runner.step_count)
    send(dict(action=ExperimentProtocol.METADATA, payload=metadata))
    recv()


def main():
    app, bc, args = make_remote_app()
    app.run(break_condition=bc)


if __name__ == "__main__":
    main()
