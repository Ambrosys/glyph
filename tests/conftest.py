import pytest
import glyph.gp as gp


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")


@pytest.fixture(scope="function")
def SympyIndividual():
    return gp.Individual(pset=gp.sympy_primitive_set(constants=["c"]), name="SympyIndividual")


@pytest.fixture(scope="function")
def NumpyIndividual():
    # 13 for the boston data set
    return gp.Individual(gp.numpy_primitive_set(arity=13), "NumpyIndividual")


@pytest.fixture(scope="function", params=[SympyIndividual, NumpyIndividual])
def IndividualClass(request):
    cls = request.param
    return cls()


@pytest.fixture(scope="function", params=gp.all_algorithms)
def AlgorithmClass(request):
    cls = request.param
    return cls
