import argparse
import collections
import json
import logging

from experiment import EventLoop


class DummyEventLoop(EventLoop):
    def __init__(self, config):
        super().__init__(None, config)

    def evaluate(self, pop):
        self.logger.debug(len(pop))
        return dict(fitness=[0 for _ in pop])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None)

    args = parser.parse_args()
    if args.file:
        with open(args.file, "r") as f:
            cfile = json.load(f)
    else:
        cfile = {}

    primitives = {"x": 0, "k0": -1, "k1": -1, "Add": 2, "Mul": 2, "Sub": 2}
    defaults = {"primitives": primitives}

    config = collections.ChainMap(cfile, defaults)

    logging.basicConfig(level=logging.DEBUG)

    loop = DummyEventLoop(dict(config))
    loop.run()
