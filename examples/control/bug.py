import os
import sys

sys.path.append(os.path.dirname(__file__))

import numpy as np
from scipy.integrate import odeint

from glyph import gp
import control_problem


def dy(x, t):
    return -0.2*x**3 + x

x0 = 1
t = np.linspace(0, 10, 100)

x = odeint(dy, x0, t)[:, 0]


print("done")

def test(actuator):
    def dy(t, y, *args):
        dx = - 0.2*y**3 + actuator(*y, *args)
        return dx
    return dy



class Individual(gp.AExpressionTree):
    """The gp representation (genotype) of the actuator for the control problem."""

    pset = gp.sympy_primitive_set(categories=['algebraic', 'trigonometric', 'exponential'],
                                  arguments=['x_0'], constants=['c'])

expr = "Mul(c, x_0)"
ind = Individual.from_string(expr)
ind.popt = [1]

func = gp.sympy_phenotype(ind)
dy = test(func)

x_glyph = control_problem.integrate(dy, x0, t, f_args=ind.popt)

for rtol in [1e-2, 1e-3, 1e-4, 1e-5]:
    np.testing.assert_allclose(x, x_glyph, rtol=rtol)

import matplotlib.pyplot as plt
plt.plot(t, x)
plt.plot(t, x_glyph)
plt.show()
