import argparse
import collections

import zmq
import json
import logging

import numpy as np
from deap.gp import compile

from build_pset import build_pset

logger = logging.getLogger(__name__)


class EventLoop(object):
    def __init__(self, experiment, config, socket=None, port=5555):
        self.socket = socket or zmq.Context().socket(zmq.REP)
        self.port = port
        self.experiment = experiment
        self.config = config
        self.pset = build_pset(config["primitives"])

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
        elif action == "METADATA":
            logger.info(request['payload'])
            return ""
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
    def __init__(self, arity):

        def target(x):
            return np.array([f(x) for f in [lambda x: 1.2*x[:, 0]**2, lambda x: 0.3*x[:, 0] + 1.1]])

        self.x = np.random.random(size=(200, arity))
        self.y = target(self.x)

        self.metric = lambda y, yhat: np.sum((y - yhat)**2)

    def __call__(self, funcs):
        yhat = [f(*self.x.T) for f in funcs]
        for i in range(len(yhat)):
            if np.isscalar(yhat[i]):
                yhat[i] = np.ones_like(self.y[i]) * yhat[i]
        yhat = np.array(yhat)
        return self.metric(self.y, yhat)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None)

    args = parser.parse_args()
    if args.file:
        with open(args.file, "r") as f:
            cfile = json.load(f)
    else:
        cfile = {}

    primitives = {"x": 0, "k0":-1, "k1": -1, "Add": 2, "Mul": 2, "Sub": 2}
    defaults = {"primitives": primitives}

    config = collections.ChainMap(cfile, defaults)

    logging.basicConfig(level=logging.DEBUG)
    experiment = Experiment(len([v for v in config["primitives"].values() if v == 0]))

    loop = EventLoop(experiment, dict(config))
    loop.run()
