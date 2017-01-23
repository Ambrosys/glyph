import time

import pytest

from glyph.utils.timeout import *


def test_SoftTimeOut():
    ttl = 1
    sttl = SoftTimeOut(ttl)

    assert sttl.alive
    time.sleep(ttl + 1)

    assert not sttl()


def test_timeout_decorator():
    ttl = 2

    @timeout(ttl)
    def long_running_function(s):
        time.sleep(s)
        return True

    with pytest.raises(TimeoutError):
        long_running_function(ttl + 1)

    assert long_running_function(ttl - 1)


def test_max_fitness_on_timeout():
    f = lambda: 1
    def g():
        raise TimeoutError

    decorator = max_fitness_on_timeout(2)

    assert decorator(f)() == 1
    assert decorator(g)() == 2
