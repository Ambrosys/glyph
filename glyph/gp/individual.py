"""Provide Individual class for gp."""

import sys
import abc
import deap.gp
import deap.base
import sympy
import functools
import itertools
import numpy as np


def sympy_primitive_set(categories=('algebraic', 'trigonometric', 'exponential'), arguments=['y_0'], constants=[], arity=0):
    """Create a primitive set with sympy primitves.

    :param categories: an optional list of function categories for the primitive set. The following are available
                       'algebraic', 'trigonometric', 'exponential', 'exponential', 'logarithm', 'sqrt'.
    """
    pset = deap.gp.PrimitiveSet('main', arity=arity)
    if 'algebraic' in categories:
        pset.addPrimitive(sympy.Add, arity=2)
        pset.addPrimitive(sympy.Mul, arity=2)
        pset.addPrimitive(functools.partial(sympy.Mul, -1.0), arity=1, name='Neg')
    if 'trigonometric' in categories:
        pset.addPrimitive(sympy.sin, arity=1)
        pset.addPrimitive(sympy.cos, arity=1)
    if 'exponential' in categories:
        pset.addPrimitive(sympy.exp, arity=1)
    if 'logarithm' in categories:
        pset.addPrimitive(sympy.ln, arity=1)
    if 'sqrt' in categories:
        pset.addPrimitive(sympy.sqrt, arity=1)

    # Use sympy symbols for argument representation.
    for symbol in itertools.chain(arguments, constants):
        pset.addTerminal(sympy.Symbol(symbol), name=symbol)
    # Dirty hack to make constant optimization possible.
    pset.args = arguments
    pset.consts = constants
    return pset


def sympy_phenotype(individual):
    """Compile python function from individual.

    Uses sympy's lambdify function. Terminals from the primitive set will be
    used as parameters to the constructed lambda function; primitives (like
    sympy.exp) will be converted into numpy expressions (eg. numpy.exp).
    """
    # args = ','.join(terminal.name for terminal in individual.terminals)
    pset = individual.pset
    args = ','.join(arg for arg in itertools.chain(pset.args, pset.consts))
    expr = sympy.sympify(individual.compile())
    func = sympy.utilities.lambdify(args, expr, modules='numpy')
    return func


class Symc(deap.gp.Ephemeral):
    func = staticmethod(lambda: 1.0)
    ret = deap.gp.__type__


def numpy_primitive_set(arity, categories=('algebraic', 'trigonometric', 'exponential', 'symc')):

    pset = deap.gp.PrimitiveSet("main", arity)
    # Use primitive set built-in for argument representation.
    pset.renameArguments(**{'ARG{}'.format(i): 'x{}'.format(i) for i in range(arity)})
    if 'symc' in categories:
        pset._add(Symc)
        pset.terms_count += 1

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

    if 'algebraic' in categories:
        pset.addPrimitive(np.add, 2, name="Add")
        pset.addPrimitive(np.subtract, 2, name="Sub")
        pset.addPrimitive(np.multiply, 2, name="Mul")
        _div = close_function(np.divide, 1)
        pset.addPrimitive(_div, 2, name="Div")
    if 'trigonometric' in categories:
        pset.addPrimitive(np.sin, arity=1)
        pset.addPrimitive(np.cos, arity=1)
    if 'exponential' in categories:
        pset.addPrimitive(np.exp, arity=1)
        _log = close_function(np.log, 1)
        pset.addPrimitive(_log, arity=1)
    if 'sqrt' in categories:
        _sqrt = close_function(np.sqrt, 1)
        pset.addPrimitive(_sqrt, arity=1)
    if "constants" in categories:
        pset.addTerminal(np.pi, name="pi")
        pset.addTerminal(np.e, name="e")

    nan_context = {'nan': np.nan, 'inf': np.inf}
    pset.context.update(nan_context)

    return pset


class AExpressionTree(deap.gp.PrimitiveTree):
    """Abstract base class for the genotype.

    Derived classes need to specify a primitive set from which the expression
    tree can be build, as well as a phenotype method.
    """

    pset = None
    hasher = str

    def __init__(self, content):
        if self.pset is None:
            raise TypeError("Cannot instantiate abstract {} with abstract attribute pset.".format(self.__class__))
        super(AExpressionTree, self).__init__(content)
        self.fitness = Measure()

    def __repr__(self):
        """Symbolic representation of the expression tree."""
        repr = ''
        stack = []
        for node in self:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                repr = prim.format(*args)
                if len(stack) == 0:
                    break
                stack[-1][1].append(repr)
        return repr

    def __hash__(self):
        return hash(self.hasher(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def compile(self):
        """Compile the individual's representation into python code.

        Code taken from deap.gp.compile.

        :returns: either a lambda function if the primitive set has one or more
                  arguments defined, or just the evluated function body
                  otherwise.
        """
        code = repr(self)
        if len(self.pset.arguments) > 0:
            args = ",".join(arg for arg in self.pset.arguments)
            code = "lambda {}: {}".format(args, code)
        try:
            return eval(code, self.pset.context, {})
        except MemoryError:
            _, _, traceback = sys.exc_info()
            raise MemoryError('Error in tree evaluation : '
                              'Python cannot evaluate a tree higher than 90.').with_traceback(traceback)

    @property
    def terminals(self):
        """Return terminals that occur in the expression tree."""
        return [primitive for primitive in self if primitive.arity == 0]

    @classmethod
    def from_string(cls, string):
        return super(AExpressionTree, cls).from_string(string, cls.pset)

    @classmethod
    def create_population(cls, size, gen_method=deap.gp.genHalfAndHalf, min=1, max=4):
        """Create a list of individuals of class Individual."""
        if size < 0:
            raise RuntimeError('Cannot create population of size {}'.format(size))
        toolbox = deap.base.Toolbox()
        toolbox.register("expr", gen_method, pset=cls.pset, min_=min, max_=max)
        toolbox.register("individual", deap.tools.initIterate, cls, toolbox.expr)
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
        return toolbox.population(n=size)


class ANDimTree(list, metaclass=abc.ABCMeta):

    def __init__(self, trees):
        super().__init__(trees)
        self.dim = len(trees)
        self.fitness = Measure()

    @abc.abstractproperty
    def base(self):
        pass

    def compile(self):
        funcs = [tree.compile() for tree in self]
        return lambda *x: [func(*x) for func in funcs]

    def __repr__(self):
        return str([str(tree) for tree in self])

    @classmethod
    def create_individual(cls, ndim):
        return cls(cls.base.create_population(ndim))

    @classmethod
    def create_population(cls, size, ndim):
        return [cls.create_individual(ndim) for _ in range(size)]

    @property
    def height(self):
        return len(self)


class Measure(deap.base.Fitness):
    """
    This is basically a wrapper around deap.base.Fitness.

    It provides the following enhancements over the base class:
    - more adequate naming
    - copy constructable
    - no weight attribute
    """

    weights = itertools.repeat(-1)

    def __init__(self, values=()):
        if len(values) > 0:
            self.values = values

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def get_values(self):
        return tuple(-1.0 * v for v in self.wvalues)

    def set_values(self, values):
        self.wvalues = tuple(-1.0 * v for v in values)

    def del_values(self):
        self.wvalues = ()

    values = property(get_values, set_values, del_values)
