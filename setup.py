import os

from setuptools import find_packages, setup

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

with open(os.path.join(here, "requirements.txt"), "r") as f:
    REQUIRED = f.readlines()

REQUIRED_GUI = ["gooey>=1.0.0"]

with open(os.path.join(here, "README.md"), "r") as f:
    LONG_DESCRIPTION = f.read()

if __name__ == "__main__":
    setup(
        name=NAME,
        version=versioneer.get_version(),
        author=AUTHOR,
        author_email=EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
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
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        extras_require={"gui": REQUIRED_GUI},
        entry_points={
            "console_scripts": [
                "glyph-remote = glyph.cli.glyph_remote:main"
            ]
        },
        cmdclass=versioneer.get_cmdclass(),
    )
