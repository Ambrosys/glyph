"""Some usefull classes/functions for the fitness assessment part in gp problems."""

import numpy as np
import scipy
import toolz
import functools
import warnings

from glyph.gp.individual import _get_index

class SingleProcessFactoy(object):
    map = map

    def __call__(self):
        return self


class AAssessmentRunner(object):
    """Abstract runner for the (parallel) assessment of individuals in a population.

    Child classes have to at least override the measure() method, which might be
    executed in a different process or even on a remote machine depending on the
    parallelization scheme.
    Child classes may override the setup() method, which is executed once on
    object instantiation.
    Child classes may override the assign_fitness() method, which is executed in
    the main process. This can be usefull if you want to locally post-process
    the results of measure(), when collected from remote processes.
    """

    def __init__(self, parallel_factory=SingleProcessFactoy()):
        """Store an instance of parallel_factory.

        :param parallel_factory: callable() -> obj, obj has to implement some
                                 kind of (parallel) map() method.
        """
        self.parallel_factory = parallel_factory
        self._parallel = None
        self.setup()

    def __call__(self, population):
        """Update the fitness of each individual in population that has an invalid fitness.

        :param population: a squence of individuals.
        """
        if self._parallel is None:
            self._parallel = self.parallel_factory()
        invalid = [p for p in population if not p.fitness.valid]
        fitnesses = self._parallel.map(self.measure, invalid)
        for ind, fit in zip(invalid, fitnesses):
            self.assign_fitness(ind, fit)
        return len(invalid)

    def __getstate__(self):
        """Modify pickling behavior for the class.

        All the attributes except 'parallel' can be pickled.
        """
        state = self.__dict__.copy()
        del state['_parallel']  # Remove the unpicklable parallelization scheme.
        return state

    def __setstate__(self, state):
        """Modify unpickling behavior for the class."""
        self.__dict__.update(state)  # Restore instance attributes.
        self._parallel = None  # Restoring the parallelization scheme happens when the functor is called.

    def setup(self):
        """Default implementation."""
        pass

    def assign_fitness(self, individual, fitness):
        """Assign a fitness value (as returned by self.measure()) to idividual.

        Default implementation.
        """
        individual.fitness.values = fitness

    def measure(self, individual):
        """Return a fitness value for individual."""
        raise NotImplementedError


def measure(*funcs, pre=toolz.identity, post=toolz.identity):
    """
    Combine several measurement functions into one.

    Optionaly do pre- and/or post-processing.

    :param funcs: a sequence of measure functions as returned by measure() (eg.
                   callable(*a, **kw) -> tuple), and/or single valued functions
                   (eg. callable(*a, **kw) -> numerical value).
    :param pre: some pre-processing function that is to be apllied on input
                *once* before passing the result to each function in *funcs.
    :param post: some post-processing function that is to be apllied on the
                 tuple of measure values as returned by the combined *funcs.
    :returns: callable(input) -> tuple of measure values, where input is usually
              a phenotype (eg. an expression tree).
    """
    def closure(*args, **kwargs) -> tuple:
        m = toolz.compose(post, _tt_flatten, toolz.juxt([tuple_wrap(f) for f in funcs]), pre)
        return m(*args, **kwargs)
    closure.size = sum(getattr(f, 'size', 1) for f in funcs)
    return closure


def default_constants(ind):
    if ind.pset.constants:
        consts_types = ind.pset.constants
        if len(consts_types) == 1 and "Symc" in consts_types:   # symc case
            values = np.ones(len(_get_index(ind, consts_types[0])))
        else:                           # sympy case
            values = np.ones(len(consts_types))
    else:
        values = []
    return values


def const_opt_scalar(measure, individual, bounds=None, method='Powell', default_constants=default_constants):
    """Apply constant optimization on a scalar measure.

    Uses scipy.optimize.minimize().

    :param measure: callable(individual, *f_args) -> scalar.
    :param individual: an individual tha is passed on to measure.
    :bounds: bounds for the constant values (s. scipy.optimize.minimize).
    :method: Type of solver. Should either be 'leastsq', or one of
             scipy.optimize.minimize's solvers.
    :returns: (popt, measure_opt), popt: the optimal values for the constants;
              measure_opt: the measure evaluated at popt.
    """
    @functools.wraps(measure)
    def closure(args):
        return measure(individual, *args)
    p0 = default_constants(individual)
    popt = p0
    measure_opt = None
    terminals = [t.name for t in individual.terminals]
    if any(constant in terminals for constant in individual.pset.constants):
        res = scipy.optimize.minimize(fun=closure, x0=p0, bounds=bounds, method=method)
        popt = res.x if res.x.shape else np.array([res.x])
        measure_opt = res.fun
        if not res.success:
            warnings.warn(res.message, UserWarning)
    if measure_opt is None:
        measure_opt = closure(popt)
    return popt, measure_opt


def const_opt_leastsq(measure, individual, default_constants=default_constants):
    """Apply constant optimization on a vector valued measure.

    Uses scipy.optimize.leastsq().

    :param measure: callable(individual, *f_args) -> numeric sequence.
    :param individual: an individual tha is passed on to measure.
    :returns: (popt, measure_opt), popt: the optimal values for the constants;
              measure_opt: the measure evaluated at popt.
    """
    @functools.wraps(measure)
    def closure(args):
        return measure(individual, *args)
    p0 = default_constants(individual)
    popt = p0
    measure_opt = None
    terminals = [t.name for t in individual.terminals]
    if any(str(constant) in terminals for constant in individual.pset.constants):
        res = scipy.optimize.leastsq(func=closure, x0=p0, full_output=True)
        popt, infodict, msg, ierr = res[0], res[2], res[-2], res[-1]
        measure_opt = infodict['fvec']
        if ierr < 0 or ierr > 4:
            warnings.warn(msg, UserWarning)
    if measure_opt is None:
        measure_opt = closure(popt)
    return popt, measure_opt


def replace_nan(x, rep=np.infty):
    """Replace occurences of np.NAN in x.

    Heads-up: Does not work on np.arrays.
    """
    try:
        return type(x)(each if not np.isnan(each) else rep for each in x)
    except TypeError:
        return x if not np.isnan(x) else rep


def _tt_flatten(tt):
    """Flatten a tuple of tuples."""
    return tuple(sum(tt, ()))


def _argcount(func):
    """Return argument count of func."""
    return func.__code__.co_argcount


def annotate(func, annotations):
    """Add annoations to func."""
    if not hasattr(func, '__annoations__'):
        setattr(func, '__annotations__', annotations)
    else:
        func.__annotations__.update(annotations)
    return func


def returns(func, type):
    """Check func's annotation dictionary for return type tuple."""
    try:
        return func.__annotations__['return'] is type
    except (AttributeError, KeyError):
        return False


def tuple_wrap(func):
    """Wrap func's return value into a tuple if it is not one already."""
    if returns(func, type=tuple):
        return func  # No need to wrap.
    else:
        @functools.wraps(func)
        def closure(*args, **kwargs):
            res = func(*args, **kwargs)
            if isinstance(res, tuple):
                return res
            else:
                return res,
        annotate(closure, {'return': tuple})
        return closure
