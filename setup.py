import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

import versioneer

NAME = "pyglyph"
DESCRIPTION = "Symbolic regression tools."
URL = "https://github.com/Ambrosys/glyph"
EMAIL = "markus.quade@ambrosys.de"
AUTHOR = "Markus Abel, Julien Gout, Markus Quade"
KEYWORDS = "complex systems, control, machine learning, genetic programming"
LICENCE = "LGPL"
PYTHON = ">=3.5"

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements-to-freeze.txt"), "r") as f:
    REQUIRED = f.readlines()

with io.open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    LONG_DESCRIPTION = "\n" + f.read()


if __name__ == "__main__":
    setup(
        name=NAME,
        version=versioneer.get_version(),
        author=AUTHOR,
        author_email=EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENCE,
        keywords=KEYWORDS,
        url=URL,
        packages=find_packages(exclude=["tests", "doc", "examples"]),
        install_requires=REQUIRED,
        python_requires=PYTHON,
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
        ],
        entry_points={
            "console_scripts": [
                "glyph-remote = glyph.cli.glyph_remote:main"
            ]
        },
        cmdclass=versioneer.get_cmdclass(),
    )
