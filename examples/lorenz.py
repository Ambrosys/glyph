"""gp application."""

import logging
import argparse
from toolz import cons

import sympy
import sympy.utilities
import numpy
import deap

import glyph.application as application
import glyph.gp as gp
import glyph.control_problem as control_problem
import glyph.assessment as assessment
import glyph.utils as utils


# Setup of the control problem and gp algorithm.
class Individual(gp.AExpressionTree):
    """The gp representation (genotype) of the actuator for the control problem."""

    pset = gp.sympy_primitive_set(categories=['algebraic', 'trigonometric', 'exponential'],
                                  arguments=['y_0', 'y_1', 'y_2'], constants=['c'])

    def __str__(self):
        """Human readable representation of the individual."""
        return str(sympy.sympify(deap.gp.compile(repr(self), self.pset)))



class AssessmentRunner(assessment.AAssessmentRunner):
    """Define a measure for the fitness assessment."""

    def setup(self):
        """Setup dynamic system."""
        self.x = numpy.linspace(0.0, 100.0, 5000, dtype=numpy.float64)
        self.yinit = numpy.array([10.0, 1.0, 5.0])
        self.params = dict(s=10.0, r=28.0, b=8.0 / 3.0)
        self.target = numpy.zeros_like(self.x)

    def measure(self, individual):
        popt, rmse_opt = assessment.const_opt_leastsq(self.rmse, individual, numpy.ones(len(individual.pset.constants)))
        return rmse_opt[0], rmse_opt[1], rmse_opt[2], len(individual), popt

    def assign_fitness(self, individual, fitness):
        individual.fitness.values = fitness[:-1]
        individual.popt = fitness[-1]

    # TODO(jg): maybe as cached (because of constant optimization).
    def rmse(self, individual, *f_args):
        y = self.trajectory(individual, *f_args)
        rmse_y_0 = utils.numeric.rmse(self.target, y[0, :])
        rmse_y_1 = utils.numeric.rmse(self.target, y[1, :])
        rmse_y_2 = utils.numeric.rmse(self.target, y[2, :])
        return assessment.replace_nan((rmse_y_0, rmse_y_1, rmse_y_2))

    def trajectory(self, individual, *f_args):
        dy = control_problem.lorenz_in_2(gp.sympy_phenotype(individual), **self.params)
        return utils.numeric.integrate(dy, yinit=self.yinit, x=self.x, f_args=f_args)


def main():
    """Entry point of application."""
    program_description = 'Lorenz system'
    parser = argparse.ArgumentParser(program_description)
    parser.add_argument('--params', type=utils.argparse.ntuple(3, float), default=(10, 28, 8/3),
                        help='parameters σ,r,b for the lorenz system (default: 10,28,8/3)')
    parser.add_argument('--plot', help='plot best results', action='store_true')

    app, args = application.default_console_app(Individual, AssessmentRunner, parser)
    app.assessment_runner.params['s'] = args.params[0]
    app.assessment_runner.params['r'] = args.params[1]
    app.assessment_runner.params['b'] = args.params[2]
    app.run()

    logger = logging.getLogger(__name__)
    logger.info('\n')
    logger.info('Hall of Fame:')
    for individual in app.gp_runner.halloffame:
        logger.info('{}  {}'.format(individual.fitness.values, str(individual)))

    if not args.plot:
        return
    # Plot n best results.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn
    n = 2
    seaborn.set_palette('husl', n + 2)
    alpha = 0.7
    label_size = 16
    title_size = 20
    params, yinit = app.assessment_runner.params, app.assessment_runner.yinit
    x, target = app.assessment_runner.x, app.assessment_runner.target
    title = program_description + '\nparams={}, yinit={}'.format(params, yinit)
    ax0 = plt.subplot2grid((3, 2), (0, 0))
    ax1 = plt.subplot2grid((3, 2), (1, 0))
    ax2 = plt.subplot2grid((3, 2), (2, 0))
    ax3 = plt.subplot2grid((3, 2), (1, 1), projection='3d', rowspan=2)
    lines, labels = [], []
    l, = ax0.plot(x, target, linestyle='dotted')
    ax1.plot(x, target, linestyle='dotted')
    ax2.plot(x, target, linestyle='dotted')
    labels.append('target')
    lines.append(l)
    uncontrolled = Individual.from_string('Add(y_0, Neg(y_0))')
    for ind in cons(uncontrolled, app.gp_runner.halloffame[:n]):
        popt = getattr(ind, 'popt', numpy.zeros(len(ind.pset.constants)))
        label = 'with $a({}) = {}$, $c={}$'.format(','.join(ind.pset.args), str(ind), popt)
        label = label.replace('**', '^').replace('*', '\cdot ')
        y = app.assessment_runner.trajectory(ind, *popt)
        l, = ax0.plot(x, y[0, :], alpha=alpha)
        ax1.plot(x, y[1, :], alpha=alpha, color=l.get_color())
        ax2.plot(x, y[2, :], alpha=alpha, color=l.get_color())
        ax3.plot(y[0, :], y[1, :], y[2, :], alpha=alpha, color=l.get_color())
        labels.append(label)
        lines.append(l)
    ax0.set_ylabel('$y_0$', fontsize=label_size)
    ax0.set_xlabel('time', fontsize=label_size)
    ax1.set_ylabel('$y_1$', fontsize=label_size)
    ax1.set_xlabel('time', fontsize=label_size)
    ax2.set_ylabel('$y_2$', fontsize=label_size)
    ax2.set_xlabel('time', fontsize=label_size)
    ax3.set_xlabel('$y_0$', fontsize=label_size)
    ax3.set_ylabel('$y_1$', fontsize=label_size)
    ax3.set_title('Phase Portrait', fontsize=label_size)
    plt.figlegend(lines, labels, loc='upper right', bbox_to_anchor=(0.9, 0.9), fontsize=label_size)
    plt.suptitle(title, fontsize=title_size)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()


if __name__ == '__main__':
    main()
