def test_hash(IndividualClass):
    ind = IndividualClass.create_population(1)[0]
    pop = [ind, ind]
    assert len(set(pop)) == 1


def test_reproducibility(IndividualClass):
    import random

    seed = 1234567890
    random.seed(seed)
    population_1 = IndividualClass.create_population(1000)
    random.seed(seed)
    population_2 = IndividualClass.create_population(1000)
    assert population_1 == population_2
