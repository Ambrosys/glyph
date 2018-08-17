import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import glyph.gp
glyph.gp.individual.np = np
from glyph.gp.individual import numpy_phenotype, numpy_primitive_set, Individual

pset = numpy_primitive_set(1)
MyIndividual = Individual(pset=pset)


def elementwise_grad(fun):
    return grad(lambda x: np.sum(fun(x)))


if __name__ == '__main__':
    expr = "Mul(x_0, Add(x_0, x_0))"
    ind = MyIndividual.from_string(expr)
    f = numpy_phenotype(ind)
    df = elementwise_grad(f)
    ddf = elementwise_grad(df)
    x = np.linspace(-1, 1, 101)

    plt.plot(x, f(x))
    plt.plot(x, df(x))
    plt.plot(x, ddf(x))
    plt.show()
