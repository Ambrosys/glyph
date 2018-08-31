# Copyright: 2017, Markus Abel, Julien Gout, Markus Quade
# Licence: LGPL

import functools
from collections import defaultdict

import numpy as np
import scipy.optimize


def rms(y):
    """Root mean square."""
    return np.sqrt(np.mean(np.square(y)))


def strict_subtract(x, y):
    try:
        if x.shape != y.shape:
            raise ValueError('operands could not be broadcast together with shapes {} {}'.format(x.shape, y.shape))
    except AttributeError:
        pass
    return x - y


def rmse(x, y):
    """Root mean square error."""
    return rms(strict_subtract(x, y))


def nrmse(x, y):
    """Normalized, with respect to x, root mean square error."""
    diff = strict_subtract(x, y)
    return rms(diff) / (np.max(x) - np.min(x))


def cvrmse(x, y):
    """Coefficient of variation, with respect to x, of the rmse."""
    diff = strict_subtract(x, y)
    return rms(diff) / np.mean(x)


def silent_numpy(func):
    @functools.wraps(func)
    def closure(*args, **kwargs):
        with np.errstate(all='ignore'):
            return func(*args, **kwargs)
    return closure


def hill_climb(fun, x0, args, precision=5, maxfev=100, directions=5, target=0, rng=np.random, **kwargs):
    """Stochastic hill climber for constant optimization.
    Try self.directions different solutions per iteration to select a new best individual.

    :param fun: function to optimize
    :param x0: initial guess
    :param args: additional arguments to pass to fun
    :param precision: maximum precision of x0
    :param maxfev: maximum number of function calls before stopping
    :param directions: number of directions to explore before doing a hill climb step
    :param target: stop if fun(x) <= target
    :param rng: (seeded) random number generator

    :return: `scipy.optimize.OptimizeResult`
    """
    res = scipy.optimize.OptimizeResult()

    def tweak(x):
        """ x = round(x + xi, p) with xi ~ N(0, sqrt(x)+10**(-p))"""
        return round(x+rng.normal(scale=np.sqrt(abs(x))+10**(-precision)), precision)

    def f(x):
        return fun(x, *args)

    x = x0
    fx = f(x)
    it = 1
    if len(x0) > 0:
        while fx >= target and it <= maxfev:
            memory = [(x, fx)]
            for j in range(directions):
                it += 1
                xtweak = np.array([tweak(c) for c in x])
                fxtweak = f(xtweak)
                memory.append((xtweak, fxtweak))
                if fxtweak <= target:
                    break
            x, fx = min(memory, key=lambda t: t[1])

    res["x"] = x
    res["fun"] = fx
    res["success"] = True
    res["nfev"] = it

    return res


class SmartConstantOptimizer:
    """Decorate a minimize method used in `scipy.optimize.minimize` to cancel non promising constant optimizations.

    The stopping criteria is based on the improvement rate :math:`\frac{\Delta f}[\Delta fev}`.

    If the improvement rate is below the :math:`q_{threshold}` quantile for a given number of function
    evaluations, optimization is stopped.
    """
    def __init__(self, method, step_size=10, min_stat=10, threshold=25):
        """
        :params method: see `scipy.optimize.minimize` method
        :params step_size: number of function evaluations betweem iterations
        :params min_stat: minimum sample size before stopping
        :params threshold: quantile
        """
        self.method = method
        self.step_size = step_size
        self.min_stat = min_stat
        self.threshold = threshold
        self.memory = defaultdict(list)

    def __call__(self, fun, x0, args, **kwargs):

        maxfev = kwargs.get('maxfev', 1000*len(x0))
        kw = kwargs.copy()
        kw["maxfev"] = self.step_size

        res = self.method(fun, x0, args, **kw)
        fx_base = res.fun
        x0 = res.x

        fev = self.step_size
        while fev <= maxfev - self.step_size:

            res = self.method(fun, x0, args, **kw)
            fev += res.nfev
            fx = res.fun
            x0 = res.x

            eps = (fx - fx_base) / fev
            self.memory[fev].append(eps)
            if len(self.memory[fev]) > self.min_stat:
                if eps < np.percentile(self.memory[fev], self.threshold):
                    break

        return res


def expressional_complexity(ind):
    """Sum of length of all subtrees of the individual."""
    return sum(len(ind[ind.searchSubtree(i)]) for i in range(len(ind)))
