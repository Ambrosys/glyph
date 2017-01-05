# Concepts

 * Individual
 * Genetic operators
 * Evolver (Algorithm)
 * AssessmentRunner
 * GPRunner
 * Application

### Individual

 * Implements representation.
 * Tree, Stack, n-dim, etc.
 * Should implement `__str__` for parsing.
 * Should implement slicing?

### Genetic operators
 * For each individual, for each dimension we have multiple operators and possible combinations, e.g. for mutation we have:
   * basic: `mutshrink`, `mutinsert`, `mutreplace`, `muterc`
   * random of one of the above
   * maybe representation-specific operators
 * Can we address different representations with the slicing operator? Or do we need specific methods for each representation.
 * Example: `ind[3]` gives the 3rd prim of a tree-based individual. But you will need to delete the corresponding subtree to create a syntactic correct offspring using `mutshrink`.
 * If so: _Multiple dispatch functions_ vs. _class methods_ (this may become an issue in memory consumption and symbolic regression. `gplearn` has this pattern and the issue.).

### Evolver (Algorithm)

 * Provides a single method `evolve(pop)`. Typically this includes:
     * Constructing the archive (e.g. using nsga2).
     * Selecting parents for breeding (e.g. tournament)
     * Breeding offsprings.

### AssessmentRunner & Parallel factory
 * (optional) Provides interface to remote machines.
 * Provide evaluation logic
 *  You can combine the Evolver and AssessmentRunner to a GPRunner like so `pops = iterate(combine(Evolver.evolve, AssessmentRunner.update_fitness), init_pop())`
 * Parallel factory: map-object for embarrassingly parallely computation

### GPRunner

 * Initializes a GP run.
 * Iterates Evolver and AssessmentRunner.
 * Handles random state.

### Application

 * Provides CLI.
 * Executes callbacks, e.g. logbook or hall of fame updates.
 * Checkpointing
 * Iterates GPRunner until breakcondition or maximum genereation reached
