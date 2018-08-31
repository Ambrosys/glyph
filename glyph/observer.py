import matplotlib.pyplot as plt
import numpy as np


def get_limits(x, factor=1.1):
    """Calculates the plot range given an array x."""
    avg = np.nanmean(x)
    l = np.nanmax(x) - np.nanmin(x)
    if l == 0:
        l = 0.5
    return avg - l/2 * factor, avg + l/2 * factor


class ProgressObserver(object):  # pragma: no cover
    def __init__(self):
        """Animates the progress of the evolutionary optimization.

        Note:
            Uses matplotlibs interactive mode.
        """
        plt.ion()
        self.fig = None
        self.axis = None
        self.lines = []
        self.t_key = "gen"

    @staticmethod
    def _update_plt(ax, line, *data):
        x, y = data
        ax.set_xlim(*get_limits(x))
        ax.set_ylim(*get_limits(y))
        line.set_data(x, y)

    def _blank_canvas(self, chapters):
        self.fig, self.axes = plt.subplots(nrows=len(chapters) + 1)
        for c, ax in zip(chapters, self.axes):
            ax.set_xlabel(self.t_key)
            ax.set_ylabel(c)
            ax.set_title("Best " + c)
            line, = ax.plot([], [])
            self.lines.append(line)
        line, = self.axes[-1].plot([], [],  "o-")
        self.axes[-1].set_xlabel(chapters[0])
        self.axes[-1].set_ylabel(chapters[1])
        self.axes[-1].set_title("Pareto Front")
        self.lines.append(line)

        plt.tight_layout()

    def __call__(self, app):
        """
        Note:
            To be used as a callback in :class:`glyph.application.Application`.

        Args:
            app (glyph.application.Application)
        """
        chapters = sorted(app.logbook.chapters.keys())

        if self.fig is None:
            self._blank_canvas(chapters)

        t = app.logbook.select(self.t_key)
        for c, ax, line in zip(chapters, self.axes, self.lines):
            self._update_plt(ax, line, t, app.logbook.chapters[c].select("min"))

        x, y = zip(*sorted([i.fitness.values for i in app.gp_runner.pareto_front]))
        self._update_plt(self.axes[-1], self.lines[-1], x, y)

        self.fig.canvas.draw()
