import time

import numpy as np


class SoftTimeOut:
    def __init__(self, ttl):
        """Break condition based on a soft time out.

        Start a new generation as long as there is some time left.

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


def soft_max_iter(app, max_iter=np.infty):
    """Soft breaking condition. Will check after each generation weather maximum number of iterations is exceeded.

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


def break_condition(target=0, error_index=0, ttl=0, max_iter=np.infty):
    """Combined breaking condition based on time to live, minimum target and maximum number of iterations.

    :param target: value of desired error metric
    :param error_index: index in fitness tuple
    :param ttl: time to live in seconds
    :param max_iter: maximum number of iterations
    """

    sto = SoftTimeOut(ttl)

    def cb(app):
        return soft_max_iter(app, max_iter=max_iter) or sto(app) or soft_target(app, target=target, error_index=error_index)
    return cb
