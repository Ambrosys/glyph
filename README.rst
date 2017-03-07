glyph - symbolic regression tools
=================================

|Build Status| |AppVeyor| |PyPI version| |codecov| |PythonVersion| |Licence| |DOI|

**glyph** is a python 3 library based on deap providing abstraction layers for symbolic regression problems.

It comes with batteries included:

- predefined primitive sets
- n-dimensional expression tree class
- symbolic and structural constants
- interfacing constant optimization to scipy.optimize
- easy integration with joblib or dask.distributed
- symbolic constraints
- boilerplate code for logging, checkpointing, break conditions and command line applications
- rich set of algorithms

glyph also includes a plug and play command line application **glyph-remote** which lets non-domain experts apply symbolic regression to their optimization tasks.

Installation
------------

Glyph is a **python 3.5+** only package.

You can install the latest stable version from PyPI with pip

``pip install pyglyph``

or get the bleeding edge

``pip install git+git://github.com/ambrosys/glyph.git#egg=glyph``

Examples
--------

Examples can be found in the
`repo <https://github.com/Ambrosys/glyph/tree/master/examples>`__. To
run them you need to:

- Clone the repo.
- ``make init``
- ``cd examples``
- Run any example, e.g. ``python lorenz.py --help``

Documentation
-------------

The online documentation is available at
`glyph.readthedocs.io <https://glyph.readthedocs.io>`__.

Bugs, feature requests, contributions
-------------------------------------

Please use the `issue
tracker <https://github.com/Ambrosys/glyph/issues>`__ and the `mailing
list <https://groups.google.com/forum/#!forum/pyglyph>`__. For
contributions have a look at out `contribution
guide <https://github.com/ambrosys/glyph/blob/master/.github/CONTRIBUTING>`__.

.. |Build Status| image:: https://travis-ci.org/Ambrosys/glyph.svg?branch=master
   :target: https://travis-ci.org/Ambrosys/glyph
.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/rbl2b44yfnfk4owi/branch/master?svg=true
   :target: https://ci.appveyor.com/project/Ohjeah/glyph
.. |PyPI version| image:: https://badge.fury.io/py/pyglyph.svg
   :target: https://badge.fury.io/py/pyglyph
.. |codecov| image:: https://codecov.io/gh/Ambrosys/glyph/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/Ambrosys/glyph
.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/pyglyph.svg
   :target: https://img.shields.io/pypi/pyversions/pyglyph.svg
.. |Licence| image:: https://img.shields.io/pypi/l/pyglyph.svg
   :target: https://img.shields.io/pypi/l/pyglyph.svg
.. |DOI| image:: https://zenodo.org/badge/75950324.svg
   :target: https://zenodo.org/badge/latestdoi/75950324
