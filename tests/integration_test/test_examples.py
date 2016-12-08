import pytest
import os
import inspect
import glob


THIS_FILES_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
EXAMPLE_FOLDER_PATTERN = THIS_FILES_DIR + "/../../examples/*.py"


@pytest.mark.parametrize("file", glob.glob(EXAMPLE_FOLDER_PATTERN))
def test_example(file):
    error_code = os.system("python {}".format(file))
    assert error_code == 0
