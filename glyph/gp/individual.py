# Copyright: 2017, Markus Abel, Julien Gout, Markus Quade
# Licence: LGPL


"""Provide Individual class for gp."""

import abc
import copy
import functools
import itertools
import re
import sys

import deap.base
import deap.gp
import numpy as np
import sympy

import glyph.utils


#def len_subtree(i):
#    sl_left = ind.searchSubtree(i+1)
#    len_left = sl_left.stop - sl_left.start
#    sl_right = ind.searchSubtree(sl_left.stop)
#    len_right = sl_right.stop - sl_right.start
#    return len_left, len_right


def sc_qout(x, y):
    """SC is the quotient of the number of nodes of its left and right child-trees x and y"""
    return x / y


def sc_mmqout(x, y, cmin=-1, cmax=1):
    """SC is the minimum-maximum quotient of the number of nodes of both
    child-trees x and y mapped into the constant interval [cmin, cmax]"""
    return cmin + min(x, y)/max(x, y) * (cmax - cmin)


class StructConst(deap.gp.Primitive):
    def __init__(self, func):
        """
        :param func: evaluate left and right subtree and assign a constant.
        """
        self.func = func
        super().__init__("SC", [deap.gp.__type__]*2, deap.gp.__type__)

    @staticmethod
    def get_len(expr, tokens=("(,")):
        regex = "|".join("\\{}".format(t) for t in tokens)
        return len(re.split(regex, expr))

    def format(self, *args):
        left, right = args
        return str(self.func(self.get_len(left), self.get_len(right)))


def add_sc(pset, func):
    """Adds a structural constant to a given primitive set.
    :param func: `callable(x, y) -> float` where x and y are the expressions of the left and right subtree
    :param pset: You may want to use `sympy_primitive_set` or `numpy_primitive_set` without symbolic constants.
    :type pset: `deap.gp.PrimitiveSet`
    """
    sc = StructConst(func)
    pset._add(sc)
    pset.prims_count += 1
    return pset


def _build_args_string(pset, consts):
    args = ','.join(arg for arg in pset.args)
    if consts:
        if args:
            args += ','
        args += ','.join("{}=1.0".format(arg) for arg in consts)
    return args


def sympy_primitive_set(categories=('algebraic', 'trigonometric', 'exponential'), arguments=['y_0'], constants=[]):
    """Create a primitive set with sympy primitves.

    :param arguments: variables to use in primitive set
    :param constants: symbolic constants to use in primitive set
    :param categories: an optional list of function categories for the primitive set. The following are available
                       'algebraic', 'neg', 'trigonometric', 'exponential', 'exponential', 'logarithm', 'sqrt'.

     :return: `deap.gp.PrimitiveSet`
    """
    pset = deap.gp.PrimitiveSet('main', arity=0)
    if 'algebraic' in categories:
        pset.addPrimitive(sympy.Add, arity=2)
        pset.addPrimitive(sympy.Mul, arity=2)
    if 'neg' in categories:
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

    pset.args = arguments
    pset.constants = constants
    return pset


def sympy_phenotype(individual):
    """Compile python function from individual.

    Uses sympy's lambdify function. Terminals from the primitive set will be
    used as parameters to the constructed lambda function; primitives (like
    sympy.exp) will be converted into numpy expressions (eg. numpy.exp).

    :param individual:
    :type individual: `glyph.gp.individual.AExpressionTree`
    :return: lambda function
    """
    pset = individual.pset
    args = _build_args_string(pset, pset.constants)
    expr = sympy.sympify(deap.gp.compile(repr(individual), pset))
    func = sympy.utilities.lambdify(args, expr, modules='numpy')
    return func


def numpy_primitive_set(arity, categories=('algebraic', 'trigonometric', 'exponential', 'symc')):
    """Create a primitive set based on numpys vectorized functions.


    :param arity: Number of variables in the primitive set
    :param categories:
    :return: `deap.gp.PrimitiveSet`

    .. note :: All functions will be closed, that is non-defined values will be mapped to 1. 1/0 = 1!
    """
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
        @glyph.utils.numeric.silent_numpy
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
    """ Lambdify the individual

    :param individual:
    :type individual: `glyph.gp.individual.AExpressionTree`
    :return: lambda function

    :Note:
    In constrast to sympy_phenotype the callable will have a variable number of keyword arguments depending on the number
    of symbolic constants in the individual.

    :Example:
    >>> pset = numpy_primitive_set(1)
    >>> MyIndividual = Individual(pset=pset)
    >>> ind = MyIndividual.from_string("Add(x_0, Symc)")
    >>> f = numpy_phenotype(ind)
    >>> f(1, 1)
    2
    """
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


class Individual(type):
    """Metaclass to set the primitive set in ExpressionTree types."""

    def __new__(mcs, pset, name="MyIndividual", **kwargs):
        """Construct a new expression tree type.

        Args:
            pset: :class:`deap.gp.PrimitiveSet`
            name: name of the expression tree class
            kwargs: additional attributes

        Returns: expression tree class
        """
        cls = super().__new__(mcs, name, (AExpressionTree,), dict(pset=pset, **kwargs))
        setattr(sys.modules[__name__], name, cls)
        return cls

    def __init__(cls, pset, name="MyIndividual", **kwargs):  # noqa
        super().__init__(name, (AExpressionTree,), dict(pset=pset, **kwargs))


class AExpressionTree(deap.gp.PrimitiveTree):
    hasher = str

    def __init__(self, content):
        """Abstract base class for the genotype.

        Derived classes need to specify a primitive set from which the expression
        tree can be build, as well as a phenotype method.
        """
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
    def const_opt(self):
        return getattr(self, "popt", ())

    @property
    def terminals(self):
        """Return terminals that occur in the expression tree."""
        return [primitive for primitive in self if primitive.arity == 0]

    @property
    @abc.abstractmethod
    def pset(self):
        pass

    @classmethod
    def from_string(cls, string):
        return super(AExpressionTree, cls).from_string(string, cls.pset)

    @classmethod
    def create(cls, gen_method=deap.gp.genHalfAndHalf, min=1, max=4):
        return cls(gen_method(cls.pset, min_=min, max_=max))

    @classmethod
    def create_population(cls, size, gen_method=deap.gp.genHalfAndHalf, min=1, max=4):
        """Create a list of individuals of class Individual."""
        if size < 0:
            raise RuntimeError('Cannot create population of size {}'.format(size))
        return [cls.create(gen_method=gen_method, min=min, max=max) for _ in range(size)]


def nd_phenotype(nd_tree, backend=sympy_phenotype):
    """
    :param nd_tree:
    :type  nd_tree: ANDimTree
    :param backend: sympy_phenotype or numpy_phenotype
    :return: lambda function
    """
    funcs = [backend(t) for t in nd_tree]
    return lambda *x: [f(*x) for f in funcs]


class NDIndividual(type):
    def __new__(mcs, base, name="MyNDIndividual", **kwargs):
        cls = super().__new__(mcs, name, (ANDimTree,), dict(base=base, **kwargs))
        setattr(sys.modules[__name__], name, cls)
        return cls

    def __init__(cls, base, name="MyNDIndividual", **kwargs):  # noqa
        """Construct a new n-dimensional expression tree type.

        Args:
            base (Individual): one dimensional base class
            name: name of the n-dimensional expression tree class
            **kwargs: addtional attributes

        Returns: n-dimensional expression tree class
        """
        super().__init__(name, (ANDimTree,), dict(base=base, **kwargs))


class ANDimTree(list):
    """ A naive tree class representing a vector-valued expression.

    Each dimension is encoded as a expression tree.
    """
    def __init__(self, trees):
        super().__init__(trees)
        self.dim = len(trees)
        self.fitness = Measure()

    @property
    @abc.abstractmethod
    def base(self):
        pass

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

    @property
    def pset(self):
        return self.base.pset

    @property
    def terminals(self):
        """Return terminals that occur in the expression tree."""
        return [primitive for primitive in itertools.chain.from_iterable(self) if primitive.arity == 0]

    @classmethod
    def from_string(cls, strs):
        return cls([cls.base.from_string(s) for s in strs])


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
    prim = copy.copy(prim)
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


@glyph.utils.Memoize
def simplify_this(expr):
    """
    :param expr:
    :type expr: str
    :return: Sympy representation of simplified expression#

    :warning: does not respect closures
    """
    with glyph.utils.random_state(simplify_this):  # to avoid strange side effects of sympy testcases and random
        return sympy.simplify(stringify_for_sympy(expr))


def child_trees(ind):
    start = 1
    while start < len(ind):
        slice_ = ind.searchSubtree(start)
        yield type(ind)(ind[slice_])
        start += slice_.stop - 1


@glyph.utils.Memoize
def simplify_constant(ind):
    """Trims subtrees of symbolic constants.

    Recursively applying these rules:

        S f(*[symc]*f.arity) = symc
        S f(x,... symc) = f(x,... symc)
        S symc = symc

    For deeper trees, try to trim down lower levels first.
    An individual cannot be trimmed down further if its a fixpoint of S.
    """
    symc = ind.pset.mapping[ind.pset.constants[0]]
    if symc in ind:
        root = ind.root

        # the tree is just a function of constants
        if len(ind) > 1 and  all(i == symc for i in ind[1:]):
            return type(ind)([symc])

        # root is just a constant or tree is a function of a variable and cannot be trimmed down.
        elif len(ind) == 1 or len(ind) == root.arity + 1:
            return ind

        # try to simplify all children of root
        else:
            acc = [simplify_constant(child) for child in child_trees(ind)]
            new_ind = type(ind)([root] + sum(acc, []))

            # cannot be simplified
            if ind == new_ind:
                return ind

            # can it be simplified further?
            else:
                return simplify_constant(new_ind)

    else:
        return ind


def _constant_normal_form(expr, variables=()):
    """experimental
    """
    if expr.is_constant(*variables):
        return sympy.Symbol("c")

    args = expr.args
    if not args:
        return expr
    elif isinstance(expr, sympy.Pow):
        return type(expr)(*[_constant_normal_form(a, variables=variables) for a in args[:-1]] + [args[-1]])
    else:
        res = type(expr)(*[_constant_normal_form(a, variables=variables) for a in args])
        if res == expr:
            return expr
        else:
            return _constant_normal_form(res, variables=variables)


def pretty_print(expr, constants, consts_values, count=0):
    """Replace symbolic constants in the str representation of an individual
    by their numeric values.

    This checks for
        - c followed by ")" or ","
        - c followed by infix operators
        - c
    """
    for k, v in zip(constants, consts_values):
        c = str(k)
        p1 = r"{c}(?=[,)])".format(c=c)
        p2 = r"^{c}$".format(c=c)
        p3 = r"(?<=[*+-/]){c}|(?<=[*+-/] ){c}|{c}(?=\s?[*+-/])".format(c=c)
        pattern = r"|".join((p1, p2, p3))
        expr = re.sub(pattern, str(v), expr, count=count)
    return expr
