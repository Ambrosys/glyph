glyph - symbolic regression tools
=================================

|Build Status| |PyPI version| |codecov| |PythonVersion| |Licence| |DOI| |PUB|

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

.. warning::
    While fully usable, glyph is still pre-1.0 software and has **no** backwards compatibility guarantees until the 1.0 release occurs!

Content
-------

.. toctree::
   :maxdepth: 1

   usr/getting_started.rst
   usr/concepts.rst
   usr/glyph_remote.rst

   Publications <usr/publications.rst>
   About <usr/about.rst>

Tutorials
+++++++++

.. toctree::
   :maxdepth: 1

   usr/examples/index.rst
   usr/tutorials/parallel.rst
   usr/tutorials/labview.rst
   usr/tutorials/matlab.rst

Development:
++++++++++++
Know what you're looking for & just need API details? View our auto-generated API documentation:

.. toctree::
   :maxdepth: 1

   API Documentation <dev/api/modules>

.. |Build Status| image:: https://travis-ci.org/Ambrosys/glyph.svg?branch=master
  :target: https://travis-ci.org/Ambrosys/glyph
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
.. |PUB| image:: https://img.shields.io/badge/DOI-10.5334%2Fjors.192-blue.svg
  :target: http://doi.org/10.5334/jors.192
