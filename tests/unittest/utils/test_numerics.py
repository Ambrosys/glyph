import pytest

from glyph.utils.numeric import *


ec_cases = (
    ("x_0", 1),
    ("Add(exp(x_0), x_0)", 8),
)


@pytest.mark.parametrize("case", ec_cases)
def test_expressional_complexity(NumpyIndividual, case):
    expr, res = case
    assert expressional_complexity(NumpyIndividual.from_string(expr)) == res
