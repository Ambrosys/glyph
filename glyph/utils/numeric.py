import itertools
import functools

import toolz
import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.signal


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


def hill_climb(fun, x0, args, **options):
    """Stochastic hill climber for constant optimization.
    Try self.directions different solutions per iteration to select a new best individual.
    This iterates self.max_steps times.
    """
    rng = options.get("rng", np.random)
    precision = options.get("precision", 5)
    max_iter = options.get("max_iter", 5)
    directions = options.get("directions", 5)
    target = options.get("target", 0)

    res = scipy.optimize.OptimizeResult()

    def tweak(x):
        """ x = round(x + xi, p) with xi ~ N(0, sqrt(x)+10**(-p))
        """
        return round(x+rng.normal(scale=np.sqrt(abs(x))+10**(-precision)), precision)

    def f(x):
        return fun(x, *args)

    x = x0
    fx = f(x)
    it = 1
    if len(x0) > 0:
        while fx >= target and it <= max_iter:
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
    res["nit"] = it

    return res
