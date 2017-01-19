import pytest
import os
import inspect
import glob
import subprocess

slow = pytest.mark.skipif(not pytest.config.getoption("--runslow"), reason="need --runslow option to run")

THIS_FILES_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
EXAMPLE_FOLDER_PATTERN = THIS_FILES_DIR + "/../../examples/*.py"


@pytest.mark.parametrize("file", glob.glob(EXAMPLE_FOLDER_PATTERN))
@slow
def test_example(file):
    error_code = os.system("python {} -n 1 -p 4".format(file))
    assert error_code == 0


def test_run_glyph_remote():
    example = os.path.abspath(os.path.join(THIS_FILES_DIR, "../../examples/remote/experiment.py"))
    exp = subprocess.Popen("python {}".format(example), shell=True)
    gp = subprocess.Popen("glyph-remote --remote --ndim 2 -n 1 -p 4", shell=True)

    gp.wait()  # gp sends shutdown to exp process
    exp.wait()
    assert exp.returncode == 0
    assert gp.returncode == 0
