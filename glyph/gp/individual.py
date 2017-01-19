"""Provide Individual class for gp."""

import re
import deap.gp
import deap.base
import sympy
import functools
import itertools
import numpy as np


def _build_args_string(pset, consts):
    args = ','.join(arg for arg in pset.args)
    if consts:
        if args:
            args += ','
        args += ','.join("{}=1.0".format(arg) for arg in consts)
    return args


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
    pset.constants = constants
    return pset


def sympy_phenotype(individual):
    """Compile python function from individual.

    Uses sympy's lambdify function. Terminals from the primitive set will be
    used as parameters to the constructed lambda function; primitives (like
    sympy.exp) will be converted into numpy expressions (eg. numpy.exp).
    """
    # args = ','.join(terminal.name for terminal in individual.terminals)
    pset = individual.pset
    args = _build_args_string(pset, pset.constants)
    expr = sympy.sympify(deap.gp.compile(repr(individual), pset))
    func = sympy.utilities.lambdify(args, expr, modules='numpy')
    return func


def numpy_primitive_set(arity, categories=('algebraic', 'trigonometric', 'exponential', 'symc')):

    pset = deap.gp.PrimitiveSet("main", arity)
    # Use primitive set built-in for argument representation.
    pset.renameArguments(**{'ARG{}'.format(i): 'x_{}'.format(i) for i in range(arity)})
    pset.args = pset.arguments
    if 'symc' in categories:
        symc = 1.0
        pset.addTerminal(symc, "Symc")
        pset.constants = ["Symc"]
    else:
        pset.constants = []

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
    return pset


def _get_index(ind, c):
    return [i for i, node in enumerate(ind) if node.name == c]


def numpy_phenotype(individual):
    pset = individual.pset
    if pset.constants:
        c = pset.constants[0]
        index = _get_index(individual, c)
    else:
        index = []
    consts = ["c_{}".format(i) for i in range(len(index))]
    args = _build_args_string(pset, consts)
    expr = repr(individual)
    for c_ in consts:
        expr = expr.replace(c, c_, 1)
    func = sympy.utilities.lambdify(args, expr, modules=pset.context)
    return func


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


class ANDimTree(list):

    def __init__(self, trees):
        super().__init__(trees)
        self.dim = len(trees)
        self.fitness = Measure()

    @property
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


def convert_inverse_prim(prim, args):
    """
    Convert inverse prims according to:
    [Dd]iv(a,b) -> Mul[a, 1/b]
    [Ss]ub(a,b) -> Add[a, -b]

    We achieve this by overwriting the corresponding format method of the sub and div prim.
    """

    prim.name = re.sub(r'([A-Z])', lambda pat: pat.group(1).lower(), prim.name)    # lower all capital letters

    converter = {
        'sub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        'div': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_)
    }
    prim_formatter = converter.get(prim.name, prim.format)

    return prim_formatter(*args)


def stringify_for_sympy(f):
    """Return the expression in a human readable string.
    """
    string = ""
    stack = []
    for node in f:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = convert_inverse_prim(prim, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string


def simplify_this(expr):
    return sympy.simplify(stringify_for_sympy(expr))
