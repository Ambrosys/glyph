import zmq
import json

if __name__ == '__main__':
    socket = zmq.Context.instance().socket(zmq.REP)
    socket.bind("tcp://*:5557")

    def send(msg):
        socket.send(json.dumps(msg).encode('ascii'))

    while True:
        print(json.loads(socket.recv().decode('ascii')))
        send(dict(fitness=[0]))
