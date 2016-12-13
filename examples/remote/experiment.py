import zmq
import json
import logging

import numpy as np
from deap.gp import compile
from control.gp import numpy_primitive_set

from build_pset import build_pset


class EventLoop(object):
    def __init__(self, experiment, primitives, socket=None, port=5555):
        self.socket = socket or zmq.Context().socket(zmq.REP)
        self.port = port
        self.experiment = experiment
        self.primitives = primitives
        self.pset = build_pset(primitives)

    @property
    def config(self):
        return dict(primitives=self.primitives, pop_size=10, generations=5, consider_complexity=False)

    @property
    def address(self):
        return "tcp://*:{}".format(self.port)

    def start(self):
        self.socket.bind(self.address)

    def shutdown(self):
        self.socket.close()

    def run(self):
        self.start()
        while True:
            request = json.loads(self.socket.recv().decode('ascii'))
            logger.info(request)
            result = self.work(request)
            logger.info(result)
            if result is None:
                break
            self.socket.send(json.dumps(result).encode('ascii'))

    def work(self, request):
        action = request['action']
        if action == "CONFIG":
            return self.config
        elif action == "SHUTDOWN":
            return self.shutdown()
        elif action == "EXPERIMENT":
            return self.evaluate(request['payload'])
        else:
            raise ValueError("Unknown action")

    def evaluate(self, individual):
        func = compile(individual, self.pset)
        fitness = self.experiment(func),
        return dict(fitness=fitness)


class Experiment(object):
    def __init__(self):
        self.points = np.linspace(-1, 1, 100, endpoint=True)
        f = lambda x: x**2 + 0.25
        self.y = f(self.points)
        self.metric = lambda y, yhat: np.sum((y - yhat)**2)

    def __call__(self, func):
        yhat = func(self.points)
        return self.metric(self.y, yhat)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    primitives = dict(x=0, add=2, multiply=2, subtract=2, negative=1)
    experiment = Experiment()

    loop = EventLoop(experiment, primitives)
    loop.run()
