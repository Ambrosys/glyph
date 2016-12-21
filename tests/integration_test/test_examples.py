import pytest
import os
import inspect
import glob
import subprocess


THIS_FILES_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
EXAMPLE_FOLDER_PATTERN = THIS_FILES_DIR + "/../../examples/*.py"


@pytest.mark.parametrize("file", glob.glob(EXAMPLE_FOLDER_PATTERN))
def test_example(file):
    error_code = os.system("python {}".format(file))
    assert error_code == 0


def test_run_remote():
    example = os.path.join(THIS_FILES_DIR, "../../examples/remote/experiment.py")
    exp = subprocess.Popen("python {}".format(example))
    gp = subprocess.Popen("glyph-remote")

    gp.wait()  # gp sends shutdown to exp process
    exp.wait()
    assert exp.returncode == 0
    assert gp.returncode == 0
