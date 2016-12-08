import dill
from dill import Pickler
import joblib
joblib.parallel.pickle = dill
joblib.pool.dumps = dill.dumps
joblib.pool.Pickler = Pickler

from joblib.pool import CustomizablePicklingQueue
from io import BytesIO
from pickle import HIGHEST_PROTOCOL


class CustomizablePickler(Pickler):
    """Pickler that accepts custom reducers.
    HIGHEST_PROTOCOL is selected by default as this pickler is used
    to pickle ephemeral datastructures for interprocess communication
    hence no backward compatibility is required.
    `reducers` is expected expected to be a dictionary with key/values
    being `(type, callable)` pairs where `callable` is a function that
    give an instance of `type` will return a tuple `(constructor,
    tuple_of_objects)` to rebuild an instance out of the pickled
    `tuple_of_objects` as would return a `__reduce__` method. See the
    standard library documentation on pickling for more details.
    """

    # We override the pure Python pickler as its the only way to be able to
    # customize the dispatch table without side effects in Python 2.6
    # to 3.2. For Python 3.3+ leverage the new dispatch_table
    # feature from http://bugs.python.org/issue14166 that makes it possible
    # to use the C implementation of the Pickler which is faster.

    def __init__(self, writer, reducers=None, protocol=HIGHEST_PROTOCOL):
        Pickler.__init__(self, writer, protocol=protocol)
        if reducers is None:
            reducers = {}
        # Make the dispatch registry an instance level attribute instead of
        # a reference to the class dictionary under Python 2
        self.dispatch = Pickler.dispatch.copy()
        for type, reduce_func in reducers.items():
            self.register(type, reduce_func)

    def register(self, type, reduce_func):
        if hasattr(Pickler, 'dispatch'):
            # Python 2 pickler dispatching is not explicitly customizable.
            # Let us use a closure to workaround this limitation.
            def dispatcher(self, obj):
                reduced = reduce_func(obj)
                self.save_reduce(obj=obj, *reduced)
            self.dispatch[type] = dispatcher
        else:
            self.dispatch_table[type] = reduce_func

joblib.pool.CustomizablePickler = CustomizablePickler


def _make_methods(self):
    self._recv = recv = self._reader.recv
    racquire, rrelease = self._rlock.acquire, self._rlock.release

    def get():
        racquire()
        try:
            return recv()
        finally:
            rrelease()

    self.get = get

    def send(obj):
        buffer = BytesIO()
        CustomizablePickler(buffer, self._reducers).dump(obj)
        self._writer.send_bytes(buffer.getvalue())

    self._send = send

    if self._wlock is None:
        # writes to a message oriented win32 pipe are atomic
        self.put = send
    else:
        wlock_acquire, wlock_release = (
            self._wlock.acquire, self._wlock.release)

        def put(obj):
            wlock_acquire()
            try:
                return send(obj)
            finally:
                wlock_release()

        self.put = put

CustomizablePicklingQueue._make_methods = _make_methods

from joblib import Parallel, delayed, parallel_backend
from distributed.joblib import DaskDistributedBackend

joblib.register_parallel_backend('dask.distributed', DaskDistributedBackend)

from glyph import application
import sys
import os
sys.path.append(os.path.dirname(__file__))
from minimal_example import complete_measure, Individual, pop_size


def update_fitness(population):
    invalid = [p for p in population if not p.fitness.valid]
    with parallel_backend('dask.distributed', scheduler_host='127.0.0.1:8786'):
        fitnesses = Parallel(n_jobs=-1)(delayed(complete_measure)(ind) for ind in invalid)
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit
    return len(invalid)


def main():

    runner = application.default_gprunner(Individual, update_fitness, algorithm='nsga2', mating='cxonepoint', mutation='mutuniform')
    runner.init(pop_size=pop_size)
    for gen in range(10):
        runner.step()
        print(runner.logbook.stream)
    for individual in runner.halloffame:
        print(individual)

if __name__ == "__main__":
    from distributed import Client
    client = Client()
    client.upload_file(os.path.join(os.path.dirname(__file__), "minimal_example.py"))
    main()
