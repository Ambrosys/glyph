glyph - symbolic regression tools
=================================

[![Build
Status](https://travis-ci.org/Ambrosys/glyph.svg?branch=master)](https://travis-ci.org/Ambrosys/glyph)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/rbl2b44yfnfk4owi/branch/master?svg=true)](https://ci.appveyor.com/project/Ohjeah/glyph)
[![Documentation Status](https://readthedocs.org/projects/glyph/badge/?version=latest)](http://glyph.readthedocs.io/en/latest/?badge=latest)
[![PyPI
version](https://img.shields.io/pypi/v/pyglyph.svg)](https://pypi.python.org/pypi/pyglyph/)
[![codecov](https://img.shields.io/codecov/c/github/Ambrosys/glyph.svg?branch=master)](https://codecov.io/gh/Ambrosys/glyph)

[![PythonVersion](https://img.shields.io/pypi/pyversions/pyglyph.svg)](https://img.shields.io/pypi/pyversions/pyglyph.svg)
[![Licence](https://img.shields.io/pypi/l/pyglyph.svg)](https://img.shields.io/pypi/l/pyglyph.svg)
[![DOI](https://zenodo.org/badge/75950324.svg)](https://zenodo.org/badge/latestdoi/75950324)

**glyph** is a python 3 library based on deap providing abstraction
layers for symbolic regression problems.

It comes with batteries included:

- predefined primitive sets
- n-dimensional expression tree class
- symbolic and structure-based constants
- interfacing constant optimization to `scipy.optimize`
- easy integration with `joblib` or `dask.distributed`
- symbolic constraints
- boilerplate code for logging, checkpointing, break conditions and command line applications
- rich set of algorithms

glyph also includes a plug and play command line application
**glyph-remote** which lets non-domain experts apply symbolic regression
to their optimization tasks.

Installation
------------

Glyph is a **python 3.6+** only package.

You can install the latest stable version from PyPI with pip

`pip install pyglyph`

or get the bleeding edge

`pip install git+git://github.com/ambrosys/glyph.git#egg=glyph`

Documentation
-------------

The online documentation is available at
[glyph.readthedocs.io](https://glyph.readthedocs.io).

Bugs, feature requests, contributions
-------------------------------------

Please use the [issue tracker](https://github.com/Ambrosys/glyph/issues).
For contributions have a look at out [contribution
guide](https://github.com/ambrosys/glyph/blob/master/.github/CONTRIBUTING).
