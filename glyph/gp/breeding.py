import toolz
import deap.gp
import itertools
import numpy as np


def mutuniform(pset, **kwargs):
    """Factory for mutuniform
    """
    min_ = kwargs.get('min_', 0)
    max_ = kwargs.get('max_', 2)
    expr_mut = toolz.partial(deap.gp.genFull, min_=min_, max_=max_)
    mutate = toolz.partial(deap.gp.mutUniform, expr=expr_mut, pset=pset)
    return mutate


def mutnodereplacement(pset, **kwargs):
    """Factory for mutnodereplacement"""
    return toolz.partial(deap.gp.mutNodeReplacement, pset=pset)


def mutinsert(pset, **kwargs):
    """Factory for mutinsert"""
    return toolz.partial(deap.gp.mutInsert, pset=pset)


def mutshrink(pset, **kwargs):
    """Factory for mutshrink"""
    return deap.gp.mutShrink


def cxonepoint(**kwargs):
    """Factory for cxonepoint"""
    return deap.gp.cxOnePoint


def cxonepointleafbiased(**kwargs):
    """Factory for cxonepointleafbiased"""
    termpb = kwargs.get('termpb', 0.1)
    return toolz.partial(deap.gp.cxOnePointLeafBiased, termpb=termpb)


def nd_mutation(atree, mut1d, rng=np.random):
    """

    :param atree:
    :type atree: `glyph.gp.individual.ANDimTree`
    :param mut1d: any mutation operator worklng for `glyph.gp.individual.AExpressionTree`
    :param rng: (seeded) random number generator
    :return: mutated offspring
    """
    a = rng.randint(0, atree.dim)
    atree[a] = mut1d(atree[a])[0]
    return atree,


def nd_crossover(atree, btree, cx1d, rng=np.random):
    """

    :param atree:
    :type atree: `glyph.gp.individual.ANDimTree`
    :param btree:
    :type btree: `glyph.gp.individual.ANDimTree`
    :param cx1d: any crossover operator working for `glyph.gp.individual.AExpressionTree`
    :param rng: (seeded) random number generator
    :return: two mated offsprings
    """
    n = atree.dim
    a, b = rng.randint(0, n, size=2)

    atree[a], btree[b] = cx1d(atree[a], btree[b])

    return atree, btree


all_mutations = [mutuniform, mutnodereplacement, mutshrink, mutshrink]
all_crossover = [cxonepoint, cxonepointleafbiased]
__all__ = [obj.__name__ for obj in itertools.chain(all_mutations, all_crossover)]
