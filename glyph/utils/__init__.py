# Copyright: 2017, Markus Abel, Julien Gout, Markus Quade
# Licence: LGPL

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
