# Copyright: 2017, Markus Abel, Julien Gout, Markus Quade
# Licence: LGPL

__path__ = __import__('pkgutil').extend_path(__path__, __name__) # namespace package

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
