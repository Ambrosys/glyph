import zmq
import json

if __name__ == '__main__':
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5557")

    def send(msg):
        socket.send(json.dumps(msg).encode('ascii'))

    experiment = dict(action="EXPERIMENT", payload="nothing")
    shutdown = dict(action="SHUTDOWN")

    for i in range(10):
        send(experiment)
        print(json.loads(socket.recv().decode('ascii')))
    send(shutdown)
