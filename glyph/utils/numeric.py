import itertools
import functools
import toolz
import numpy
import scipy.integrate
import scipy.optimize
import scipy.signal
import deap.gp


@toolz.curry
def row(n, array):
    """Take the nth row from array."""
    return array[n, :]


def integrate(dy, yinit, x, f_args=(), integrator='dopri5', **integrator_args):
    """
    Convenience function for odeint().

    Uselful if you do not want to step through the integration, but rather get
    the full result in one call.

    :param dy: callable(x, y, *args)
    :param yinit: sequence of initial values.
    :param x: sequence of x values.
    :param *f_args: (optional) extra arguments to pass to function.
    :returns: y(x)
    """
    res = odeint(dy, yinit, x, f_args=f_args, integrator=integrator, **integrator_args)
    y = numpy.vstack(res).T
    if y.shape[1] != x.shape[0]:
        y = numpy.empty((y.shape[0], x.shape[0]))
        y[:] = numpy.NAN
    if y.shape[0] == 1:
        return y[0, :]
    return y


def odeint(dy, yinit, x, f_args=(), integrator='dopri5', **integrator_args):
    """
    Integrate the initial value problem (dy, yinit).

    A wrapper around scipy.integrate.ode.

    :param dy: callable(x, y, *args)
    :param yinit: sequence of initial values.
    :param x: sequence of x values.
    :param *f_args: (optional) extra arguments to pass to function.
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
    return numpy.sqrt(numpy.mean(numpy.square(y)))


def strict_subtract(x, y):
    try:
        if x.shape != y.shape:
            raise ValueError('operands could not be broadcast together with shapes {} {}'.format(x.shape, y.shape))
    except AttributeError:
        pass
    return x - y


@toolz.curry
def rmse(x, y):
    """Root mean square error."""
    return rms(strict_subtract(x, y))


@toolz.curry
def nrmse(x, y):
    """Normalized, with respect to x, root mean square error."""
    diff = strict_subtract(x, y)
    return rms(diff) / (numpy.max(x) - numpy.min(x))


@toolz.curry
def cvrmse(x, y):
    """Coefficient of variation, with respect to x, of the rmse."""
    diff = strict_subtract(x, y)
    return rms(diff) / numpy.mean(x)


def silent_numpy(func):
    @functools.wraps(func)
    def closure(*args, **kwargs):
        with numpy.errstate(all='ignore'):
            return func(*args, **kwargs)
    return closure


# this is some legacy code to optimize constants

def generate_context(pset, data):
    context = {arg: dat for arg, dat in zip(pset.arguments, data.T)}
    context.update(pset.context)

    return context


def optimize_constants(ind, cost, context, precision=3, options=None, constraints=None):
    """ Update the constant values of ind according to:
    vec(c) = argmin_c ||yhat(data,c) - y||

    This needs to be called together when using symbolic constants.
    It may be called as a mutation operator together with the usage of ercs.
    """
    idx = [index for index, node in enumerate(ind) if isinstance(node, deap.gp.Ephemeral)]

    if len(idx) == 0:
        return ind

    values = [ind[i].value for i in idx]
    args = [("c%i" % i) for i in range(len(idx))]

    code = str(ind)
    for i, arg in zip(idx, args):
        code = code.replace(ind[i].format(), arg, 1)
    code = "lambda {args}: {code}".format(args=",".join(args), code=code)
    yhat = eval(code, context, {})
    with numpy.errstate(invalid='ignore', over='ignore'):
        res = scipy.optimize.minimize(cost, values, args=yhat, options=options, constraints=constraints)

    if res.success and all(numpy.isfinite(res.x)):
        values = res.x

    for i, value in zip(idx, values):
        ind[i] = type(ind[i])()   # re-initialize the class, else it will overwrite the parents values too!
        ind[i].value = round(value, precision)

        del ind.fitness.values

    return ind
