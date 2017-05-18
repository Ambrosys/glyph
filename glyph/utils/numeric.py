# Copyright: 2017, Markus Abel, Julien Gout, Markus Quade
# Licence: LGPL

import itertools
import functools
from collections import defaultdict

import numpy as np
import scipy.integrate
import scipy.optimize


def integrate(dy, yinit, x, f_args=(), integrator='dopri5', **integrator_args):
    """
    Convenience function for odeint().

    Uselful if you do not want to step through the integration, but rather get
    the full result in one call.

    :param dy: `callable(x, y, *args)`
    :param yinit: sequence of initial values.
    :param x: sequence of x values.
    :param f_args: (optional) extra arguments to pass to function.
    :returns: y(x)
    """
    res = odeint(dy, yinit, x, f_args=f_args, integrator=integrator, **integrator_args)
    y = np.vstack(res).T
    if y.shape[1] != x.shape[0]:
        y = np.empty((y.shape[0], x.shape[0]))
        y[:] = np.NAN
    if y.shape[0] == 1:
        return y[0, :]
    return y


def odeint(dy, yinit, x, f_args=(), integrator='dopri5', **integrator_args):
    """
    Integrate the initial value problem (dy, yinit).

    A wrapper around scipy.integrate.ode.

    :param dy: `callable(x, y, *args)`
    :param yinit: sequence of initial values.
    :param x: sequence of x values.
    :param f_args: (optional) extra arguments to pass to function.
    :yields: y(x_i)
    """
    @functools.wraps(dy)
    def f(x, y):
        return dy(x, y, *f_args)
    ode = scipy.integrate.ode(f)
    ode.set_integrator(integrator, **integrator_args)
    # ode.set_f_params(*f_args)
    # Seems to be buggy! (see https://github.com/scipy/scipy/issues/1976#issuecomment-17026489)
    ode.set_initial_value(yinit, x[0])
    yield ode.y
    for value in itertools.islice(x, 1, None):
        ode.integrate(value)
        if not ode.successful():
            raise StopIteration
        yield ode.y


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

    The stopping criteria is based on the improvment rate :math:`\frac{\Delta f}[\Delta fev}`.

    If the improvment rate is below the :math:`q_{threshold}` quantile for a given number of function
    evaluations, optimization is stopped.
    """
    def __init__(self, method, step_size=10, min_stat=10, threshold=25):
        """
        :params method: see `scipy.optimize.minimize` method
        :params step_size: number of function evaluations betweem iterations
        :params min_stat: minmum sample size before stopping
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
