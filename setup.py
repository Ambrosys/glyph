"""Solving control problems with machine learning methods."""

import re
import ast
from setuptools import setup, find_packages


_version_re = re.compile(r'__VERSION__\s+=\s+(.*)')

with open('glyph/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(f.read().decode('utf-8')).group(1)))


def read(fname, split=True):
    with open(fname, 'r') as f:
        content = f.read()
    return content.split('\n') if split else content


setup(
    name='pyglyph',
    version=version,
    author='Markus Abel, Julien Gout, Markus Quade',
    author_email='markus.abel@ambrosys.de, julien.gout@ambrosys.de, markus.quade@ambrosys.de',
    description=__doc__.split('\n'),
    long_description=read('README.md', split=False),
    license=read('LICENCE', split=False),
    keywords='complex systems, control, machine learning, genetic programming',
    url='https://www.github.com/ambrosys/glyph',
    packages=find_packages(exclude=["tests", "doc", "examples"]),
    install_requires=read('requirements-to-freeze.txt'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
    ],
)
