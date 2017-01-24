"""Collection of dynamic system creation classes for control problems."""

import numpy as np
import toolz
import networkx


# Simple Systems

@toolz.curry
def harmonic_oscillator(actuator, omega=1.0):
    """Return harmonic oscillator with actuator built in.

    :param actuator: callable(*y, *args).
    :param omega: angular frequency of the oscillator.
    """
    def dy(t, y, *args):
        dy0 = y[1]
        dy1 = -omega**2 * y[0] + actuator(*y, *args)
        return [dy0, dy1]
    return dy


@toolz.curry
def anharmonic_oscillator(actuator, omega=1.0, c=1.0, k=1.0):
    """Return anharmonic oscillator with actuator built in."""
    def dy(t, y, *args):
        dy0 = y[1]
        dy1 = - omega**2 * y[0] - k * y[0]**2 - c * y[1] + actuator(*y, *args)
        return [dy0, dy1]
    return dy


@toolz.curry
def lorenz_in_3(actuator, s=10.0, r=28.0, b=8.0 / 3.0):
    """Return lorenz attractor with actuator built in."""
    # TODO(jg): How to apply actuator to dynamic system?
    def dy(t, y, *args):
        dy0 = s * (y[1] - y[0])
        dy1 = r * y[0] - y[1] - y[0] * y[2]
        dy2 = y[0] * y[1] - b * y[2] + actuator(*y, *args)
        return [dy0, dy1, dy2]
    return dy


@toolz.curry
def lorenz_in_2(actuator, s=10.0, r=28.0, b=8.0 / 3.0):
    """Return lorenz attractor with actuator built in."""
    # TODO(jg): How to apply actuator to dynamic system?
    def dy(t, y, *args):
        dy0 = s * (y[1] - y[0])
        dy1 = r * y[0] - y[1] - y[0] * y[2] + actuator(*y, *args)
        dy2 = y[0] * y[1] - b * y[2]
        return [dy0, dy1, dy2]
    return dy


# Coupled Systems

@toolz.curry
def van_der_pol(actuator, sensor=toolz.identity, omega=1.0, a=0.1, b=0.01, A=0.0):
    """Return Van der Pol oscillator with actuator built in."""
    def dy(t, y, *args):
        N = int(len(y) / 2)
        y0, y1 = y[:N], y[N:]
        dy0 = y1
        dy1 = - omega**2 * y0 + a * y1 * (1 - b * y0**2) + A.dot(y1) + actuator(*sensor(y), *args)
        return np.hstack((dy0, dy1))
    return dy


# TODO(jg): untested
@toolz.curry
def fitzhugh_nagumo(actuator, sensor=toolz.identity, a=0.7, b=0.8, tau=12.5, A=0.0):
    """Return FitzHugh-Nagumo oscillator with actuator built in."""
    if tau == 0.0:
        raise RuntimeError('Division by zero for tau = {}'.format(tau))

    def dy(t, y, *args):
        N = int(len(y) / 2)
        y0, y1 = y[:N], y[N:]
        dy0 = y0 - y0**3 / 3.0 - y1
        dy1 = (y0 + a - b * y1) / tau + A.dot(y1) + actuator(*sensor(y), *args)
        return np.hstack((dy0, dy1))
    return dy


# TODO(jg): untested
@toolz.curry
def hindmarsh_rose(actuator, sensor=toolz.identity, a=1.0, b=3.0, c=1.0, d=5.0, r=1e-3, s=4.0, xR=-8.0/5.0, A=0.0):
    """Return Hindmarsh-Rose oscillator with actuator built in."""
    def dy(t, y, *args):
        N = int(len(y) / 3)
        y0, y1, y2 = y[:N], y[N:2*N], y[2*N:]
        dy0 = y1 - a * y0**3 + b * y0**2 - y2
        dy1 = c - d * y0**2 - y1 + A.dot(y1) + actuator(*sensor(y), *args)
        dy2 = r * (s * (y0 - xR) - y2)
        return np.hstack((dy0, dy1, dy2))
    return dy


def global_coupling(N):
    """Generate a coupling matrix for global coupling.

    N = 3: A = [-2  1  1]
               [ 1 -2  1]
               [ 1  1 -2]
    """
    A = np.ones((N, N))
    np.fill_diagonal(A, -1.0 * float(N - 1))
    return A


def pairwise_coupling(N):
    """Generate a coupling matrix for pairwise coupling.

    N = 2: A = [-1  1]
               [ 1 -1]
    """
    assert N % 2 == 0
    a = np.array([[-1, 1],
                     [1, -1]])
    A = np.kron(np.eye(int(N / 2)), a)
    return A


def circular_array_coupling(N):
    """Generate a coupling matrix for circular array coupling.

    N = 4: A = [-2  1  0  1]
               [ 1 -2  1  0]
               [ 0  1 -2  1]
               [ 1  0  1 -2]
    """
    g = networkx.cycle_graph(N)
    A = -1.0 * networkx.linalg.laplacian_matrix(g)
    return A


def grid_2d_coupling(n, m, periodic=False):
    """Generate a coupling matrix for a 2D grid of n*m oscillators.

    n denotes the number of rows and m the number of columns in the grid.
    """
    g = networkx.grid_2d_graph(n, m, periodic=periodic)
    A = -1.0 * networkx.linalg.laplacian_matrix(g, nodelist=sorted(g.nodes()))
    return A


def dorogovtsev_goltsev_mendes_coupling(n):
    """Generate a coupling matrix from the dorogovtsev-goltsev-mendes graph.

    n is the generation.
    """
    g = networkx.generators.dorogovtsev_goltsev_mendes_graph(n)
    A = -1.0 * networkx.linalg.laplacian_matrix(g)
    return A
