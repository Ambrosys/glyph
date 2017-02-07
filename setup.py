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
    description="Symbolic regression tools.",
    long_description=read('README.rst', split=False),
    license='LGPL',
    keywords='complex systems, control, machine learning, genetic programming',
    url='https://www.github.com/ambrosys/glyph',
    packages=find_packages(exclude=["tests", "doc", "examples"]),
    install_requires=read('requirements-to-freeze.txt'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    entry_points={
        'console_scripts': [
            'glyph-remote = glyph.cli.glyph_remote:main'
        ]
    },
)
