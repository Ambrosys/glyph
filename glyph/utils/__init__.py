# Copyright: 2017, Markus Abel, Julien Gout, Markus Quade
# Licence: LGPL

import random
from contextlib import contextmanager

from . import argparse
from . import logging
from . import numeric
from . import break_condition


class Memoize:
    """Memoize(fn) - an instance which acts like fn but memoizes its arguments
       Will only work on functions with non-mutable arguments
       http://code.activestate.com/recipes/52201/
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in self.memo:
            self.memo[key] = self.fn(*args, **kwargs)
        return self.memo[key]

@contextmanager
def random_state(obj, rng=random):
    """Do work inside this contextmanager with a random state defined by obj.

    Looks for _prev_state to seed the rng.
    On exit, it will write the current state of the rng as _tmp_state
    to the obj.

    :params obj: Any object.
    :params rng: Instance of a random number generator.
    """
    obj._tmp_state = rng.getstate()
    rng.setstate(getattr(obj, '_prev_state', rng.getstate()))
    yield
    obj._prev_state = rng.getstate()
    rng.setstate(getattr(obj, '_tmp_state', rng.getstate()))
