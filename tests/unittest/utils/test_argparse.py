import argparse

import pytest

from glyph.utils.argparse import *


def test_positive_int():
    assert 10 == positive_int("10")

    with pytest.raises(ValueError):
        positive_int("0.5")

    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("-1")


def test_non_negative_int():
    assert 10 == non_negative_int("10")

    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_int("-1")

    with pytest.raises(ValueError):
        positive_int("0.5")


def test_unit_interval():
    assert 0.5 == unit_interval("0.5")

    with pytest.raises(argparse.ArgumentTypeError):
        unit_interval("-0.1")

    with pytest.raises(ValueError):
        unit_interval("0..5")
