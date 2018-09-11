import os
import random
import itertools
from contextlib import contextmanager

import stopit

from . import argparse
from . import logging
from . import numeric
from . import break_condition


class Memoize:
    def __init__(self, fn):
        """Memoize(fn) - an instance which acts like fn but memoizes its arguments

        Will only work on functions with non-mutable arguments
        http://code.activestate.com/recipes/52201/
        """
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


def partition(pred, iterable):
    """Use a predicate to partition entries into false entries and true entries.

    >>> is_odd = lambda x: x % 2
    >>> odd, even = partition(is_odd, range(10))
    >>> list(odd)
    [0, 2, 4, 6, 8]
    """
    t1, t2 = itertools.tee(iterable)
    return itertools.filterfalse(pred, t1), filter(pred, t2)


def key_set(itr, key=hash):
    keys = map(key, itr)
    s = {k: v for k, v in zip(keys, itr)}
    return list(s.values())


_is_posix = "posix" in os.name
Timeout = stopit.SignalTimeout if _is_posix else stopit.ThreadingTimeout
timeoutable = stopit.signal_timeoutable if _is_posix else stopit.threading_timeoutable
