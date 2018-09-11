import pathlib
import sys
import logging
import importlib

from .individual import simplify_this, AExpressionTree
from glyph.utils import Timeout

logger = logging.getLogger(__name__)


class Constraint:
    def __init__(self, spaces):
        self.spaces = spaces

    def _contains(self, element):
        if not self.spaces:
            return False
        return any(element in subspace for subspace in self.spaces)

    def __contains__(self, element):
        try:
            return self._contains(element)
        except Exception as e:
            logger.debug(f"Exception was raised during constraints check: {e}.")
            return False


class NonFiniteExpression(Constraint):
    def __init__(self, zero=True, infty=True, constant=False):
        """Use sympy to check for finite expressions.

        Args:
            zero: flag to check for zero expressions
            infty: flag to check for infinite expressions
            constant:  flag to check for constant expressions
        """
        self.zero = zero
        self.infty = infty
        self.constant = constant

    def _contains(self, element):
        expr = simplify_this(element)
        if isinstance(expr, str):
            logger.debug(f"Could not simplify {element}.")
            return False
        if self.constant:
            if expr.is_constant():
                return True
            elif all(t.name in element.pset.constants for t in element.terminals):
                return True

        if self.infty:
            if "zoo" in str(expr):
                return True

        if self.zero:
            if expr.is_zero:
                return True

        return False


class PreTest(Constraint):
    def __init__(self, fn, fun="chi"):
        """Apply pre-testing to check for constraint violation.

        The python script needs to provide a callable fun(ind).

        Args:
            fn: filename of the python script.
            fun: name of the function in fn.
        """
        fn = pathlib.Path(fn)
        sys.path.append(str(fn.parent))
        try:
            mod = importlib.import_module(fn.stem)
            self.f = getattr(mod, fun)
        except (AttributeError, ImportError):
            logger.error(f"Funktion {fun} not available in {fn}")
            self.f = lambda *args: False
        finally:
            sys.path.pop()

    def _contains(self, element):
        if self.f is None:
            logger.warning("Using invalid PretestNullSpace")
        return self.f(element)


class PreTestService(Constraint):  # todo com cannot be pickled !
    def __init__(self, assessment_runner):
        self.assessment_runner = assessment_runner

    @property
    def com(self):
        return self.assessment_runner.com

    @property
    def make_str(self):
        return self.assessment_runner.make_str

    def _contains(self, element):
        payload = self.make_str(element)
        self.com.send(dict(action="PRETEST", payload=payload))
        return self.com.recv()["ok"] == "True"


def reject_constrain_violation(constraint, n_trials=30, timeout=60):
    """Create constraints decorators based on rules.

    :param constraint:
    :param n_trials: Number of tries. Give up afterwards (return input).

    :return: list of constraint decorators
    """

    def reject(operator):
        def inner(*inds, **kw):
            with Timeout(timeout) as to_ctx:
                for i in range(n_trials):
                    out = operator(*inds, **kw)
                    if isinstance(out, AExpressionTree):  # can this be done w/o type checking?
                        t = out
                    elif isinstance(out, (list, tuple)):
                        t = out[0]
                    else:
                        raise RuntimeError
                    if t not in constraint:
                        break
                else:
                    logger.warning(f"Number of tries exhausted during constrained operation {operator} on individual {inds}.")  # noqa
                    if inds:
                        return inds
                    else:
                        raise UserWarning
            if to_ctx.state == to_ctx.TIMED_OUT:
                logger.warning(f"Timeout during constrained operation {operator} on individual {inds}.")
            return out
        return inner
    return reject


def constrain(funcs, constraint, n_trials=30, timeout=60):
    """Decorate a list of genetic operators with constraints.

    :param funcs: list of operators (mate, mutate, create)
    :param constraint: instance of Nullspace
    :return: constrained operators
    """

    if not constraint:
        return funcs
    return [reject_constrain_violation(constraint, n_trials=n_trials, timeout=timeout)(f)
            for f in funcs]
