import zmq
import json
import logging

import numpy as np
from deap.gp import compile

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
        return dict(primitives=self.primitives, simplify=False)# , pop_size=20, num_generations=20, consider_complexity=True, simplify=False)

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

    def evaluate(self, pop):
        fitnesses = []
        logger.debug(len(pop))
        for ind in pop:
            func = [compile(t, self.pset) for t in ind]
            fitnesses.append(self.experiment(func))
        return dict(fitness=fitnesses)


class Experiment(object):
    def __init__(self):

        def target(x):
            return np.array([f(x) for f in [lambda x: 1.2*x**2, lambda x: 0.3*x + 1.1]])

        self.x = np.linspace(-1, 1, 30)
        self.y = target(self.x)

        self.metric = lambda y, yhat: np.sum((y - yhat)**2)

    def __call__(self, funcs):
        yhat = [f(self.x) for f in funcs]
        for i in range(len(yhat)):
            if np.isscalar(yhat[i]):
                yhat[i] = np.ones_like(self.y[i]) * yhat[i]
        yhat = np.array(yhat)
        return self.metric(self.y, yhat)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    primitives = {"x": 0, "k0":-1, "k1": -1, "Add": 2, "Mul": 2, "Sub":2 }
    experiment = Experiment()

    loop = EventLoop(experiment, primitives)
    loop.run()
