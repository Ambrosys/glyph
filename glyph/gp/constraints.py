# Copyright: 2017, Markus Abel, Julien Gout, Markus Quade
# Licence: LGPL

from sympy import Float

from .individual import simplify_this, AExpressionTree


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

            if all(t.name in element.pset.constants for t in element.terminals):
                return True

        return False


def build_constraints(null_space, n_trials=20):
    """Create constraints decorators based on rules.

    :param null_space:
    :param n_trials: Number of tries. Give up afterwards (return input).

    :return: list of constraint decorators
    """
    def reject(operator):
        def inner(*inds, **kw):
            for i in range(n_trials):
                out = operator(*inds, **kw)
                if isinstance(out, AExpressionTree):  # can this be done w/o type checking?
                    t = out
                elif isinstance(out, (list, tuple)):
                    t = out[0]
                else:
                    raise RuntimeError
                if not t in null_space:
                    break
            else:
                if inds:
                    return inds
                else:
                    raise UserWarning
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
