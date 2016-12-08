"""gp application."""

import logging
import argparse
from toolz import partial, compose, cons

import sympy
import sympy.utilities
import numpy

import glyph.application as application
import glyph.gp as gp
import glyph.control_problem as control_problem
import glyph.assessment as assessment
import glyph.utils as utils


# Setup of the control problem and gp algorithm.
class Individual(gp.AExpressionTree):
    """The gp representation (genotype) of the actuator for the control problem."""

    pset = gp.sympy_primitive_set(categories=['algebraic', 'trigonometric', 'exponential'],
                                  arguments=['y_0', 'y_1'])

    def __str__(self):
        """Human readable representation of the individual."""
        return str(sympy.sympify(self.compile()))


class AssessmentRunner(assessment.AAssessmentRunner):
    """Define a measure for the fitness assessment."""

    def setup(self):
        """Setup dynamic system."""
        self.nperiods = 10.0
        self.x = numpy.linspace(0.0, self.nperiods * 2.0 * numpy.pi, 2000, dtype=numpy.float64)
        self.yinit = numpy.array([1.0, 0.0])
        self.params = dict(omega=-1)
        self.dynsys = control_problem.harmonic_oscillator(**self.params)
        self.target = numpy.zeros_like(self.x)

    def measure(self, individual):
        y = self.trajectory(individual)
        rmse_y_0 = utils.numeric.rmse(self.target, y[0, :])
        return assessment.replace_nan((rmse_y_0, len(individual)))

    def trajectory(self, individual, *f_args):
        dy = self.dynsys(gp.sympy_phenotype(individual))
        return utils.numeric.integrate(dy, yinit=self.yinit, x=self.x, f_args=f_args)


def main():
    """Entry point of application."""
    program_description = 'Harmonic oscillator'
    parser = argparse.ArgumentParser(program_description)
    parser.add_argument('--plot', help='plot best results', action='store_true')

    app, args = application.default_console_app(Individual, AssessmentRunner, parser=parser)
    app.run()

    logger = logging.getLogger(__name__)
    logger.info('\n')
    logger.info('Hall of Fame:')
    for individual in app.gp_runner.halloffame[:4]:
        logger.info('{}  {} = {}'.format(individual.fitness.values, repr(individual), str(individual)))

    if not args.plot:
        return
    # Plot n best results.
    import matplotlib.pyplot as plt
    import seaborn
    n = 4
    seaborn.set_palette('husl', n + 2)
    alpha = 0.75
    label_size = 16
    title_size = 20
    assessment_runner = AssessmentRunner()
    params, yinit = assessment_runner.params, assessment_runner.yinit
    x, target = assessment_runner.x, assessment_runner.target
    title = program_description + '\nparams={}, yinit={}'.format(params, yinit, fontsize=title_size)
    ax0 = plt.subplot2grid((2, 2), (0, 0))
    ax1 = plt.subplot2grid((2, 2), (1, 0))
    ax2 = plt.subplot2grid((2, 2), (1, 1))
    lines, labels = [], []
    l, = ax0.plot(x, target, alpha=alpha)
    labels.append('target')
    lines.append(l)
    for ind in reversed(app.gp_runner.halloffame[:n]):
        label = 'with $a(y_0, y_1) = {}$'.format(str(ind))
        label = label.replace('**', '^').replace('*', '\cdot ')
        y = assessment_runner.trajectory(ind)
        l, = ax0.plot(x, y[0, :], alpha=alpha)
        ax1.plot(x, y[1, :], alpha=alpha, color=l.get_color())
        ax2.plot(y[0, :], y[1, :], alpha=alpha, color=l.get_color())
        labels.append(label)
        lines.append(l)
    ax0.set_ylabel('$y_0$', fontsize=label_size)
    ax0.set_xlabel('time', fontsize=label_size)
    ax1.set_ylabel('$y_1$', fontsize=label_size)
    ax1.set_xlabel('time', fontsize=label_size)
    ax2.set_xlabel('$y_0$', fontsize=label_size)
    ax2.set_ylabel('$y_1$', fontsize=label_size)
    ax2.set_title('Phase Portrait', fontsize=label_size)
    plt.figlegend(lines, labels, loc='upper right', bbox_to_anchor=(0.9, 0.9), fontsize=label_size)
    plt.suptitle(title, fontsize=title_size)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()

if __name__ == '__main__':
    main()
