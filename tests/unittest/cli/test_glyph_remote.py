from functools import partial
from threading import Thread
import pytest
import zmq


from glyph.cli.glyph_remote import *


PORT = 12345

@pytest.fixture(scope="module")
def primitives():
    prims = dict(f=2, x=0, k=-1)
    return prims


@pytest.fixture(scope="module")
def ndtree(primitives):
    pset = build_pset_gp(primitives)
    Tree = type("Tree", (Individual, ), dict(pset=pset))
    MyTree = type("MyTree", (NDTree, ), dict(base=Tree))
    return MyTree


def test_build_pset_gp(primitives):
    pset = build_pset_gp(primitives)
    assert len(pset.terminals[object]) == 2
    assert pset.constants == {"k"}


class DummyExperiment:   # a copy of our example
    def __init__(self):
        self.socket = zmq.Context().socket(zmq.REP)

    def start(self):
        address = "tcp://*:{}".format(PORT)
        self.socket.bind(address)

    def shutdown(self):
        self.socket.close()

    def run(self):
        self.start()
        while True:
            request = json.loads(self.socket.recv().decode('ascii'))
            result = self.work(request)
            if result == None:
                break
            self.socket.send(json.dumps(result).encode('ascii'))

    def work(self, request):
        action = request["action"]
        if action == "CONFIG":
            return dict(primitives=primitives())
        elif action == "EXPERIMENT":
            return [0 for _ in range(request["payload"])]
        else:
            return None


class MockQueue():
    def put(self, *args):
        pass
    def get(self, key):
        return 0

# @pytest.fixture(scope="function")
# def runner():
#     experiment = DummyExperiment()
#     thread = Thread(target=experiment.run)
#     thread.start()
#     send, recv = connect("127.0.0.1", PORT)
#     runner = RemoteAssessmentRunner(send, recv)
        # runner.queue = MockQueue()
        # runner.result_queue = runner.queue
#     yield runner
#     runner.send(dict(action="SHUTDOWN"))
#     thread.join()

@pytest.fixture(scope="function")
def runner():
    send = lambda x: None
    runner = RemoteAssessmentRunner(send, send, chunk_size=1)
    runner.queue = MockQueue()
    runner.result_queue = runner.queue
    return runner


def test_evaluate_single(runner, ndtree):
    ind = ndtree.create_population(1, 1)[0]
    error = runner.evaluate_single(ind)
    assert error == 0
