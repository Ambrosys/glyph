from functools import partialmethod

import toolz
import deap.gp
import itertools
import numpy as np

from glyph.gp.individual import ANDimTree


def mutuniform(pset, **kwargs):
    min_ = kwargs.get('min_', 0)
    max_ = kwargs.get('max_', 2)
    expr_mut = toolz.partial(deap.gp.genFull, min_=min_, max_=max_)
    mutate = toolz.partial(deap.gp.mutUniform, expr=expr_mut, pset=pset)
    return mutate


def mutnodereplacement(pset, **kwargs):
    return toolz.partial(deap.gp.mutNodeReplacement, pset=pset)


def mutinsert(pset, **kwargs):
    return toolz.partial(deap.gp.mutInsert, pset=pset)


def mutshrink(pset, **kwargs):
    return deap.gp.mutShrink


def cxonepoint(**kwargs):
    return deap.gp.cxOnePoint


def cxonepointleafbiased(**kwargs):
    termpb = kwargs.get('termpb', 0.1)
    return toolz.partial(deap.gp.cxOnePointLeafBiased, termpb=termpb)


def nd_mutation(atree, mut1d, rng=np.random):
    a = rng.randint(0, atree.dim)
    atree[a] = mut1d(atree[a])[0]
    return atree,


def nd_crossover(atree, btree, cx1d, rng=np.random):
    n = atree.dim
    a, b = rng.randint(0, n, size=2)

    atree[a], btree[b] = cx1d(atree[a], btree[b])

    return atree, btree


all_mutations = [mutuniform, mutnodereplacement, mutshrink, mutshrink]
all_crossover = [cxonepoint, cxonepointleafbiased]
__all__ = [obj.__name__ for obj in itertools.chain(all_mutations, all_crossover)]
