import deap.gp
import numpy as np
import functools


def close_function(func, value):
    @functools.wraps(func)
    def closure(*args):
        res = func(*args)
        if isinstance(res, np.ndarray):
            res[np.isinf(res)] = value
            res[np.isnan(res)] = value
        elif np.isinf(res) or np.isnan(res):
            res = value
        return res
    return closure


def build_pset(primitives):
    pset = deap.gp.PrimitiveSet('main', arity=0)
    for fname, arity in primitives.items():
        if arity > 0:
            func = getattr(np, fname)
            # func = close_function(func, 1)
            pset.addPrimitive(func, arity, name=fname)
        elif arity == 0:
            pset.addTerminal(fname, name=fname)
            pset.arguments.append(fname)
        else:
            raise ValueError("Wrong arity in primitive specification.")
    return pset


def build_pset_gp(primitives):
    pset = deap.gp.PrimitiveSet('main', arity=0)
    for fname, arity in primitives.items():
        if arity > 0:
            func = lambda *args: args
            pset.addPrimitive(func, arity, name=fname)
        elif arity == 0:
            pset.addTerminal(fname, name=fname)
            pset.arguments.append(fname)
        else:
            raise ValueError("Wrong arity in primitive specification.")
    return pset
