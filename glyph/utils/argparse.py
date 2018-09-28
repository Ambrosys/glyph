"""Collection of helper functions for arparse."""

import argparse
import logging
import os
import sys
import functools

import numpy as np

logger = logging.getLogger(__name__)


def positive_int(string):
    """Check whether string is an integer greater than 0."""
    try:
        value = int(string, base=10)
    except ValueError:
        raise argparse.ArgumentTypeError("invalid int value: '{}'".format(string))
    if value < 1:
        raise argparse.ArgumentTypeError("int value hast to be greater then 0: '{}'".format(string))
    return value


def non_negative_int(string):
    """Check whether string is an integer greater than -1."""
    try:
        value = int(string, base=10)
    except ValueError:
        raise argparse.ArgumentTypeError("invalid int value: '{}'".format(string))
    if value < 0:
        raise argparse.ArgumentTypeError("int value hast to be either 0 or greater then 0: '{}'".format(string))
    return value


def unit_interval(string):
    """Check whether string is a float in the interval [0.0, 1.0]."""
    try:
        value = float(string)
    except ValueError:
        raise argparse.ArgumentTypeError("invalid float value: '{}'".format(string))
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("float value hast to be in unit interval [0,1]: '{}'".format(string))
    return value


def ntuple(n, to_type=float):
    """Check whether string is an n-tuple."""
    def evaluate(string):
        try:
            value = tuple(to_type(val) for val in string.split(','))
            if len(value) != n:
                raise ValueError
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a {}-tuple of type {}".format(n, to_type))
        return value
    return evaluate


def readable_file(string):
    """Check weather file is readable"""
    path = os.path.abspath(string)
    try:
        with open(path, 'r'):
            pass
    except IOError:
        raise argparse.ArgumentTypeError("Must be a readable file path {}".format(path))
    return path


def readable_yaml_file(string):
    """Check weather file is a .yaml file and readable"""
    path = os.path.abspath(string)
    if not path.endswith(".yaml") and not path.endswith(".yml"):
        raise argparse.ArgumentTypeError("Must be a .yaml file {}".format(path))
    return readable_file(string)


def np_infinity_int(string):
    if string == str(np.infty):
        return np.infty
    else:
        try:
            value = int(string, base=10)
        except ValueError:
            raise argparse.ArgumentTypeError("invalid int value: '{}'".format(string))
        return value
