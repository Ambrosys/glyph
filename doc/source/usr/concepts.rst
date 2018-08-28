Concepts
========

Glyph has several abstraction layers. Not all of them are required to use.


Individual & genetic operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This wraps around the backend, which is currently deap.
In contrast to deap, the individual class has to be associated with a primitive set. This makes checkpointing and later evaluation of results easier.

This abstraction layer also allows for an interchangeable representation. We plan to support graphs and stacks in the future.

Genetric operators mutation and crossover operators.

Currently, we also rely on deaps sorting algorithms.

Creating an individual class is as simple as:

.. code-block:: python

   from glyph.gp.individual import AExpressionTree, numpy_primitive_set

   pset = numpy_primitive_set(1)

   class Individual(AExpressionTree):
       pset = pset

Here, we use the convinience function :code:`numpy_primitive_set` to create a primitive set based on categeories.


Algorithm
~~~~~~~~~

This encapsulates selecting parents and breeding offspring.

Glyph comes with the following algorithms:

- AgeFitness Pareto Optimization
- SPEA2
- NSGA2
- and the "unique" counterparts of all of the above.

Algorithms need the genetic operators. The chose to implement them as classes. You can change the default parameters by simply overwriting the corresponding attribute. All algorithms only expose a single method :code:`evolve(population)`. This assumes all individuals in the population have a valid fitness. :code:`evolve(population)` will first select the parents and then produce offspring. Both, parents and offspring will be returned by the method. By doing so, so can re-evaluate the parent generation if desired (e.g. to account for different operating conditions of an experiment).

.. code-block:: python

   from functools import partial
   import deap
   from glyph import gp

   mate = deap.gp.cxOnePoint
   expr_mut = partial(deap.gp.genFull, min_=0, max_=2)
   mutate = partial(deap.gp.mutUniform, expr=expr_mut, pset=Individual.pset)
   algorithm = gp.NSGA2(mate, mutate)



AssessmentRunner
~~~~~~~~~~~~~~~~

The AssesmentRunner is a callable which takes a list of Individuals and assigns a fitness to them.
This can be as simple as:

.. code-block:: python

   def meassure(ind):
       g = lambda x: x**2 - 1.1
       points = np.linspace(-1, 1, 100, endpoint=True)
       y = g(points)
       f = gp.individual.numpy_phenotype(ind)
       yhat = f(points)
       if np.isscalar(yhat):
           yhat = np.ones_like(y) * yhat
       return nrmse(y, yhat), len(ind)

    def update_fitness(population, map=map):
        invalid = [p for p in population if not p.fitness.valid]
        fitnesses = map(meassure, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit
        return population

:code:`update_fitness` is taken directly from the deap library. You can interface your symbolic regression problem by providing a different :code:`map` function. The recommenced solution is scoop. Why this does not work in most cases see :doc:`tutorials/parallel`. Which can be a bit cumbersome to write for more complex problems.

The glyph.assessment submodule has many out of the box solutions for boilerplate/utility code, constant optimization and integration multiprocessing/distributed frameworks.

The code above with constant optimization simply becomes:

.. code-block:: python

   class AssessmentRunner(AAssessmentRunner):
       def setup(self):
           self.points = np.linspace(-1, 1, 100, endpoint=True)
           self.g = lambda x: x**2 - 1.1
           self.y = self.g(self.points)

       def measure(self, ind):
           popt, error = const_opt_scalar(self.error, ind)
           ind.popt = popt
           return error, len(ind)

       def error(self, ind, *consts):
           f = numpy_phenotype(ind)
           yhat = f(self.points, *consts)
           return nrmse(self.y, yhat)

Algorithm and assessment runner already make up a program:

.. code-block:: python

   runner = AssessmentRunner()
   pop = Individual.create_population(lambda_)
   runner(pop)

   for i in range(generations):
        pop = runner(algorithm(pop))


GPRunner
~~~~~~~~

The GPRunner lets you conveniently steps cycle through the evolutionary algrithm whilst taken care for statistics and a hall of fame.

It's mostly syntatic sugar:

.. code-block:: python

   gp_runner = GPRunner(Individual, lambda: algorithm, AssessmentRunner())
   gp_runner.init()
   for i in range(generations):
       gp_runner.step()



Application
~~~~~~~~~~~

If you want a command line interface for all your hyper-parameters, checkpointing, ensuring random state handling on resume, as well as breaking conditions, the glyph.application submodule has you covered.

The module provides several facory classes which can dynamically expand an existing `argparse.ArgumentParser`. As a starting point, you can use the :code:`default_console_app` to create an app. You will only need a primitive set and an assessment runner as explained above.

.. code-block:: python

   parser = argparse.ArgumentParser(program_description)

   app, args = application.default_console_app(Individual, AssessmentRunner, parser)
   app.run()

For more involved applications you can inherit from the Application class. (see :download:`/../../glyph/cli/glyph_remote.py`).

We recommence having a look at the :download:`/../../examples/control/minimal_example.py` as well as the :download:`/../../examples/control/lorenz.py` example to see these concepts in action.
