import argparse
import os
import logging

import numpy as np

import glyph.application
from glyph.utils.argparse import (
    catch_and_log,
    positive_int,
    non_negative_int,
    np_infinity_int,
    readable_file,
    readable_yaml_file,
)

logger = logging.getLogger(__name__)

try:
    import gooey
    from gooey import Gooey, GooeyParser

    @Gooey(
        auto_start=False,
        advanced=True,
        encoding="utf-8",
        language="english",
        show_config=True,
        default_size=(1200, 1000),
        dump_build_config=False,
        load_build_config=None,
        monospace_display=False,
        disable_stop_button=False,
        show_stop_warning=True,
        force_stop_is_error=True,
        show_success_modal=True,
        run_validators=True,
        poll_external_updates=False,
        return_to_config=False,
        disable_progress_bar_animation=False,
        navigation="SIDEBAR",
        tabbed_groups=True,
        navigation_title="Actions",
        show_sidebar=False,
    )
    def get_gooey(prog="glyph-remote"):
        probably_fork = "site-packages" not in gooey.__file__
        logger.debug("Gooey located at {}.".format(gooey.__file__))
        if probably_fork:
            logger.warning("GUI input validators may have no effect")
        parser = GooeyParser(prog=prog)
        return parser

    GUI_AVAILABLE = True
except ImportError as e:
    logger.error(e)
    GUI_AVAILABLE = False
    GUI_UNAVAILABLE_MSG = """Could not start gui extention.
You need to install the gui extras.
Use the command 'pip install glyph[gui]' to do so."""


class GooeyOptionsArg:
    POSITIVE_INT = {
        "validator": {
            "callback": catch_and_log(positive_int),
            "message": "This is not a positive integer.",
        }
    }
    NON_NEGATIVE_INT = {
        "validator": {
            "callback": catch_and_log(non_negative_int),
            "message": "This is not a non negative integer.",
        }
    }
    READABLE_FILE = {
        "validator": {
            "callback": catch_and_log(readable_file),
            "message": "This is not a readable file.",
        }
    }


class MyGooeyMixin:
    def add_argument(self, *args, **kwargs):
        for key in ["widget", "gooey_options"]:
            if key in kwargs:
                del kwargs[key]
        super().add_argument(*args, **kwargs)

    def add_mutually_exclusive_group(self, *args, **kwargs):
        group = MutuallyExclusiveGroup(self, *args, **kwargs)
        self._mutually_exclusive_groups.append(group)
        return group

    def add_argument_group(self, *args, **kwargs):
        group = ArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group


class Parser(MyGooeyMixin, argparse.ArgumentParser):
    pass


class ArgumentGroup(MyGooeyMixin, argparse._ArgumentGroup):
    pass


class MutuallyExclusiveGroup(MyGooeyMixin, argparse._MutuallyExclusiveGroup):
    pass


def get_parser(parser=None):
    if parser is None:
        parser = Parser()
    if isinstance(parser, Parser):
        parser.add_argument("--gui", action="store_true", default=False)

    parser.add_argument(
        "--port",
        type=positive_int,
        default=5555,
        help="Port for the zeromq communication (default: 5555)",
        gooey_options={
            "validator": {
                "callback": catch_and_log(positive_int),
                "message": "This should be a positive port number in the range of 0 - 65535.",
            }
        },
    )
    parser.add_argument("--ip", type=str, default="localhost", help="IP of the client (default: localhost)")
    parser.add_argument(
        "--send_meta_data", action="store_true", default=False, help="Send metadata after each generation"
    )
    parser.add_argument(
        "--gui_output", action="store_true", default=False, help="Additional gui output (default: False)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbosity",
        action="count",
        default=3,
        help="set verbose output; raise verbosity level with -vv, -vvv, -vvvv from lv 1-3",
    )
    parser.add_argument(
        "--logging",
        "-l",
        dest="logging_config",
        type=str,
        default="logging.yaml",
        help="set config file for logging; overrides --verbose (default: logging.yaml)",
        widget="FileChooser",
    )

    config = parser.add_argument_group("config")
    group = config.add_mutually_exclusive_group()
    group.add_argument(
        "--remote",
        action="store_true",
        dest="remote",
        default=False,
        help="Request GP configs from experiment handler.",
    )
    group.add_argument(
        "--cfile",
        dest="cfile",
        type=readable_yaml_file,
        help="Read GP configs from file",
        widget="FileChooser",
        gooey_options={
            "validator": {
                "callback": catch_and_log(readable_yaml_file),
                "message": "This should be a readable .yaml file.",
            }
        },
    )

    glyph.application.Application.add_options(parser)
    cp_group = parser.add_mutually_exclusive_group()
    cp_group.add_argument("--ndim", type=positive_int, default=1, gooey_options=GooeyOptionsArg.POSITIVE_INT)
    cp_group.add_argument(
        "--resume",
        dest="resume_file",
        metavar="FILE",
        type=readable_file,
        help="continue previous run from a checkpoint file",
        widget="FileChooser",
        gooey_options=GooeyOptionsArg.READABLE_FILE,
    )
    cp_group.add_argument(
        "-o",
        dest="checkpoint_file",
        metavar="FILE",
        type=str,
        default=os.path.join(".", "checkpoint.pickle"),
        help="checkpoint to FILE (default: ./checkpoint.pickle)",
        widget="FileChooser",
    )

    glyph.application.AlgorithmFactory.add_options(parser.add_argument_group("algorithm"))
    group_breeding = parser.add_argument_group("breeding")
    glyph.application.MateFactory.add_options(group_breeding)
    glyph.application.MutateFactory.add_options(group_breeding)
    glyph.application.SelectFactory.add_options(group_breeding)
    glyph.application.CreateFactory.add_options(group_breeding)

    ass_group = parser.add_argument_group("assessment")
    ass_group.add_argument(
        "--simplify",
        action="store_true",
        default=False,
        help="Simplify expression before sending them. (default: False)",
    )
    ass_group.add_argument(
        "--consider_complexity",
        action="store_false",
        default=True,
        help="Consider the complexity of solutions for MOO (default: True)",
    )
    ass_group.add_argument(
        "--no_caching",
        dest="caching",
        action="store_false",
        default=True,
        help="Cache evaluation (default: False)",
    )
    ass_group.add_argument(
        "--persistent_caching",
        default=None,
        help="Key for persistent data base cache for caching between experiments (default: None)",
    )
    ass_group.add_argument(
        "--max_fev_const_opt",
        type=non_negative_int,
        default=100,
        help="Maximum number of function evaluations for constant optimization (default: 100)",
        gooey_options=GooeyOptionsArg.NON_NEGATIVE_INT,
    )
    ass_group.add_argument(
        "--directions",
        type=positive_int,
        default=5,
        help="Directions for the stochastic hill-climber (default: 5 only used in conjunction with --const_opt_method hill_climb)",
        gooey_options=GooeyOptionsArg.POSITIVE_INT,
    )
    ass_group.add_argument(
        "--precision",
        type=non_negative_int,
        default=3,
        help="Precision of constants (default: 3)",
        gooey_options=GooeyOptionsArg.NON_NEGATIVE_INT,
    )
    ass_group.add_argument(
        "--const_opt_method",
        choices=["hill_climb", "Nelder-Mead"],
        default="Nelder-Mead",
        help="Algorithm to optimize constants given a structure (default: Nelder-Mead)",
    )
    ass_group.add_argument(
        "--structural_constants",
        action="store_true",
        default=False,
        help="Make use of structural constants. (default: False)",
    )
    ass_group.add_argument(
        "--sc_min", type=float, default=-1, help="Minimum value of sc for scaling. (default: -1)"
    )
    ass_group.add_argument(
        "--sc_max", type=float, default=1, help="Maximum value of sc for scaling. (default: 1)"
    )
    ass_group.add_argument(
        "--smart", action="store_true", default=False, help="Use smart constant optimization. (default: False)"
    )
    ass_group.add_argument(
        "--smart_step_size",
        type=non_negative_int,
        default=10,
        help="Number of fev in iterative function optimization. (default: 10)",
        gooey_options=GooeyOptionsArg.NON_NEGATIVE_INT,
    )
    ass_group.add_argument(
        "--smart_min_stat",
        type=non_negative_int,
        default=10,
        help="Number of samples required prior to stopping (default: 10)",
        gooey_options=GooeyOptionsArg.NON_NEGATIVE_INT,
    )
    ass_group.add_argument(
        "--smart_threshold",
        type=non_negative_int,
        default=25,
        help="Quantile of improvement rate. Abort constant optimization if below (default: 25)",
        gooey_options=GooeyOptionsArg.NON_NEGATIVE_INT,
    )
    ass_group.add_argument(
        "--chunk_size",
        type=positive_int,
        default=30,
        help="Number of individuals send per single request. (default: 30)",
        gooey_options=GooeyOptionsArg.POSITIVE_INT,
    )
    ass_group.add_argument(
        "--multi_objective",
        action="store_true",
        default=False,
        help="Returned fitness is multi-objective (default: False)",
    )
    ass_group.add_argument(
        "--send_symbolic",
        action="store_true",
        default=False,
        help="Send the expression with symbolic constants (default: False)",
    )
    ass_group.add_argument(
        "--re_evaluate",
        action="store_true",
        default=False,
        help="Re-evaluate old individuals (default: False)",
    )

    break_condition = parser.add_argument_group("break condition")
    break_condition.add_argument(
        "--ttl",
        type=int,
        default=-1,
        help="Time to life (in seconds) until soft shutdown. -1 = no ttl (default: -1)",
    )
    break_condition.add_argument(
        "--target", type=float, default=0, help="Target error used in stopping criteria (default: 0)"
    )
    break_condition.add_argument(
        "--max_iter_total",
        type=np_infinity_int,
        default=np.infty,
        help="Maximum number of function evaluations (default: 'inf' [stands for np.infty])",
        gooey_options={
            "validator": {
                "callback": catch_and_log(np_infinity_int),
                "message": 'This is neither "inf" nor a natural number.',
            }
        },
    )

    constraints = parser.add_argument_group("constraints")
    constraints.add_argument(
        "--constraints_zero",
        action="store_false",
        default=True,
        help="Discard zero individuals (default: True)",
    )
    constraints.add_argument(
        "--constraints_constant",
        action="store_false",
        default=True,
        help="Discard constant individuals (default: True)",
    )
    constraints.add_argument(
        "--constraints_infty",
        action="store_false",
        default=True,
        help="Discard individuals with infinities (default: True)",
    )

    observer = parser.add_argument_group("observer")
    observer.add_argument(
        "--animate",
        action="store_true",
        default=False,
        help="Animate the progress of evolutionary optimization. (default: False)",
    )

    return parser
