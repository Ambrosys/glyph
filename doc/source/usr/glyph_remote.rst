Glyph remote
============

glyph-remote is shipped together with the glyph package. After
installation, the ``glyph-remote`` command is available at the command
line.

Concept
-------

With glyph-remote the separation between optimization method and
optimization task is made easy. glyph-remote runs multi IO symbolic
regression and sends candidate solution via ZeroMQ to an experiment
controller for assessment. Every hyper-parameter used is assessable and
fully configurable.

Overview
--------

To the right the optimization method is represented. The GP program can
be seen as a black box which is only accessible by the specified
interface. To the left a single experiment plus an event handler is
depicted. The latter glues optimization method and task together and
needs to understand the communication protocol defined in

.. figure:: ../_static/communication.png
   :alt:

Currently we use `client server sockets <http://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/client_server.html>`_ for glyph remote. **The user needs to implement the zmq.REP socket.**

Communication protocol
----------------------

The communication is encoded in json. A message is a json object with
two members:

.. code:: json

    {
        "action": "value",
        "payload": "value",
    }

The possible action values are:

+-------------------+--------------------+----------------------------+
| Action name       | Payload            | Expected return Value      |
+===================+====================+============================+
| *CONFIG*          | –                  | config settings            |
+-------------------+--------------------+----------------------------+
| *EXPERIMENT*      | list of expressions| list of fitness value(s)   |
+-------------------+--------------------+----------------------------+
| *SHUTDOWN*        | –                  | –                          |
+-------------------+--------------------+----------------------------+

The config action is performed prior to the evolutionary loop. Entering
the loop, for every discovered solution an *experiment* action will be
requested. Since most experiments have an intermediate compiling step, expressions will come in chunks. You can configure optional caching for re-discovered solutions.
The *shutdown* action will let the experiment program know
that the gp loop is finished and you can safely stop the hardware.

Config
~~~~~~

See Configuration section.

Experiment
~~~~~~~~~~

The *experiment* request expects a fitness value for each expression:

.. code:: json

    {
        "fitness": ["value0", "value1", ...],
    }


Shutdown
~~~~~~~~

You can properly shut down the experiment hardware.

Configuration
-------------

For a full list of configuration options and their default values type
``glyph-remote --help``.

All hyper-parameters and algorithms used have default values. You have
three options to set parameters:

* use the command line interface
* read from file (using ``--cfile myfile.yaml``)
* request from event handler (using ``--remote``)

At cli, options are specified using ``--key value``. The configuration
file has to be written in yaml, i.e.

.. code:: yaml

    key: value

The event handler should send back a similar json array

.. code:: json

    {
      "key": "value",
    }

It is mandatory to provide a information about the primitives you want
to use. The value of the "primitives" key is again a json/yaml list
specifying name: arity pairs. Arities greater one are functions, equal
to one are variables and -1 is reserved for symbolic constants.

.. code:: json

    {
      "primitives":
      {
        "add": 2,
        "x": 0,
      },
    }
