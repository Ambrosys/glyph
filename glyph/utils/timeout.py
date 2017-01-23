import time
import signal
import functools


class SoftTimeOut(object):
    """Break condition based on a soft time out.
    Start a new generation as long as there is some time left.
    """
    def __init__(self, ttl):
        """

        :param ttl: time to live in seconds
        """
        self.soft_ttl = ttl
        self.start = time.time()

    @property
    def now(self):
        return time.time() - self.start

    @property
    def alive(self):
        return bool(self.soft_ttl - self.now > 0)

    def __call__(self, *args, **kwargs):
        return self.alive


def timeout(ttl):
    """ Decorate a function. Will raise TimeourError if function call takes longer than the ttl.

    Vendored from ffx.

    :param ttl: time to live in seconds
    """
    def decorate(f):
        def handler(signum, frame):
            raise TimeoutError()

        @functools.wraps(f)
        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(ttl)
            try:
                result = f(*args, **kwargs)
            finally:
                signal.signal(signal.SIGALRM, old)
            signal.alarm(0)
            return result
        return new_f
    return decorate


def max_fitness_on_timeout(max_fitness):
    """ Decorate a function. Associate max_fitness with long running individuals.

    :param max_fitness: fitness of aborted individual calls.
    :returns: fitness or max_fitness
    """
    def decorate(f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            try:
                fitness = f(*args, **kwargs)
            except TimeoutError:
                fitness = max_fitness
            return fitness
        return inner
    return decorate
