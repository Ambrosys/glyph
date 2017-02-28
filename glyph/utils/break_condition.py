# Copyright: 2017, Markus Abel, Julien Gout, Markus Quade
# Licence: LGPL

import time
import signal
import functools

import numpy as np


class SoftTimeOut:
    """Break condition based on a soft time out.
    Start a new generation as long as there is some time left.
    """
    def __init__(self, ttl):
        """

        :param ttl: time to live in seconds
        """
        self.soft_ttl = ttl
        self.start = time.time()

    @property
    def now(self):
        return time.time() - self.start

    @property
    def alive(self):
        return bool(self.soft_ttl - self.now > 0) or self.soft_ttl == 0

    def __call__(self, *args, **kwargs):
        return self.alive


def timeout(ttl):
    """ Decorate a function. Will raise `TimeourError` if function call takes longer than the ttl.

    Vendored from ffx.

    :param ttl: time to live in seconds
    """
    def decorate(f):
        def handler(signum, frame):
            raise TimeoutError()

        @functools.wraps(f)
        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(ttl)
            try:
                result = f(*args, **kwargs)
            finally:
                signal.signal(signal.SIGALRM, old)
            signal.alarm(0)
            return result
        return new_f
    return decorate


def max_fitness_on_timeout(max_fitness):
    """ Decorate a function. Associate max_fitness with long running individuals.

    :param max_fitness: fitness of aborted individual calls.
    :returns: fitness or max_fitness
    """
    def decorate(f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            try:
                fitness = f(*args, **kwargs)
            except TimeoutError:
                fitness = max_fitness
            return fitness
        return inner
    return decorate


def soft_max_iter(app, max_iter=np.infty):
    """
    Soft breaking condition. Will check after each generation weather maximum number of iterations is exceeded.

    :type app: `glyph.application.Application`
    :param max_iter: maximum number of function evaluations
    :return: bool(iter) > max_iter
    """
    return sum(app.gp_runner.logbook.select("evals")) >= max_iter


def soft_target(app, target=0, error_index=0):
    """Soft breaking condition. Will check after each generation minimum error is reached.

    :type app: `glyph.application.Application`
    :param target: value of desired error metric
    :param error_index: index in fitness tuple
    :return: bool(min_error) <= target
    """
    return app.gp_runner.logbook.chapters["fit{}".format(error_index)].select("min")[-1] <= target


class BreakCondition:
    """Combined breaking condition based on time to live, minimum target and maximum number of iterations.

     :Example:
     >>> app = glyph.application.Application(...)
     >>> bc = BreakCondition(...)
     >>> app.run(breaking_condition=bc)
    """
    def __init__(self, target=0, error_index=0, ttl=0, max_iter=np.infty):
        """
        :param target: value of desired error metric
        :param error_index: index in fitness tuple
        :param ttl: time to live in seconds
        :param max_iter: maximum number of iterations
        """
        self.target = target
        self.error_index = error_index
        self.sto = SoftTimeOut(ttl)
        self.max_iter = max_iter

    def __call__(self, app):
        return soft_max_iter(app, max_iter=self.max_iter) or self.sto(app) or soft_target(app, target=self.target, error_index=self.error_index)
