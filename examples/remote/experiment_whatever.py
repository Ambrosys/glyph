import zmq
import json
import logging

import numpy as np
from deap.gp import compile

from build_pset import build_pset


class EventLoop(object):
    def __init__(self, primitives, socket=None, port=5555):
        self.socket = socket or zmq.Context().socket(zmq.REP)
        self.port = port
        self.primitives = primitives
        self.pset = build_pset(primitives)

    @property
    def config(self):
        return dict(primitives=self.primitives, simplify=True, send_symbolic=True , pop_size=4, num_generations=100, consider_complexity=False)

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
        return dict(fitness=[0 for _ in pop])


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    primitives = {"g": 0, "a02": -1, "Add": 2, "Mul": 2, "a01":-1, "f": 0 }

    loop = EventLoop(primitives=primitives)
    loop.run()
