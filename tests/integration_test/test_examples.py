import os
import inspect
import glob
import subprocess
import contextlib
import shutil
import tempfile

import pytest

slow = pytest.mark.skipif(not pytest.config.getoption("--runslow"), reason="need --runslow option to run")

THIS_FILES_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
EXAMPLE_FOLDER_PATTERN = THIS_FILES_DIR + "/../../examples/*.py"


@contextlib.contextmanager
def cd(newdir, cleanup=lambda: True):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
        cleanup()

@contextlib.contextmanager
def tempdir():
    dirpath = tempfile.mkdtemp()
    def cleanup():
        shutil.rmtree(dirpath)
    with cd(dirpath, cleanup):
        yield dirpath


@pytest.mark.parametrize("file", glob.glob(EXAMPLE_FOLDER_PATTERN))
@slow
def test_example(file):
    with tempdir() as dirpath:
        error_code = os.system("python {} -n 1 -p 4".format(file))
        assert error_code == 0


@pytest.mark.timeout(300)
def test_glyph_remote():
    with tempdir() as dirpath:
        example = os.path.abspath(os.path.join(THIS_FILES_DIR, "../../examples/remote/experiment.py"))
        exp = subprocess.Popen("python {}".format(example), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        gp = subprocess.Popen("glyph-remote --remote --ndim 2 -n 2 -p 4 --max_iter_total 1", shell=True)

        gp.wait()  # gp sends shutdown to exp process
        exp.wait()

        exp = subprocess.Popen("python {}".format(example), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        gp = subprocess.Popen("glyph-remote --resume --remote", shell=True)

        gp.wait()  # gp sends shutdown to exp process
        exp.wait()
        assert exp.returncode == 0
        assert gp.returncode == 0
