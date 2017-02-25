# Copyright: 2017, Markus Abel, Julien Gout, Markus Quade
# Licence: LGPL

from functools import wraps

from sympy import Float

from .individual import simplify_this


class NullSpace:    # todo documentation
    def __init__(self, zero=True, infty=True, constant=False):
        self.zero = zero
        self.infty = infty
        self.constant = constant

    def __contains__(self, element):
        expr = simplify_this(element)
        sexpr = str(expr)

        if self.infty:
            if "zoo" in str(sexpr):
                return True

        if self.zero:
            if sexpr == "0":
                return  True

        if self.constant:
            if isinstance(expr, Float):
                return True

        return False


def build_constraints(null_space, n_trials=10):
    """Create constraints decorators based on rules.

    :param null_space:
    :param n_trials: Number of tries. Give up afterwards (raise RuntimeWarning).

    :return: list of constraint decorators
    """
    def reject(operator):
        #@wraps(operator)
        def inner(*inds):
            for i in range(n_trials):
                out = operator(*inds)
                if not out in null_space:
                    break
            else:
                raise RuntimeWarning("Individual after {} trials still in null space".format(out))
            return out
        return inner
    return [reject]


def apply_constraints(funcs, constraints):
    """Decorate a list of genetic operators with constraints.

    :param funcs: list of operators (mate, mutate, create)
    :param constraints: list of constraint decorators
    :return: constrained operators
    """
    for c in constraints:
        swap = [c(func) for func in funcs]
        funcs = swap
    return funcs
