import pytest

from glyph.utils.break_condition import *

@pytest.mark.parametrize("ttl", [0, 1])
def test_SoftTimeOut(ttl):
    sttl = SoftTimeOut(ttl)

    assert sttl.alive
    time.sleep(ttl + 1)

    assert bool(ttl) != bool(sttl())


@pytest.mark.skipif(sys.platform == 'win32',
                    reason="does not run on windows")
def test_timeout_decorator():
    ttl = 2

    @timeout(ttl)
    def long_running_function(s):
        time.sleep(s)
        return True

    with pytest.raises(TimeoutError):
        long_running_function(ttl + 1)

    assert long_running_function(ttl - 1)


@pytest.mark.skipif(sys.platform == 'win32',
                    reason="does not run on windows")
def test_max_fitness_on_timeout():
    f = lambda: 1
    def g():
        raise TimeoutError

    decorator = max_fitness_on_timeout(2)

    assert decorator(f)() == 1
    assert decorator(g)() == 2
