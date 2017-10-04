import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

NAME = "pyglyph"
DESCRIPTION = "Symbolic regression tools."
URL = "https://github.com/Ambrosys/glyph"
EMAIL = "markus.quade@ambrosys.de"
AUTHOR = "Markus Abel, Julien Gout, Markus Quade"
KEYWORDS = "complex systems, control, machine learning, genetic programming"
LICENCE = "LGPL"

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements-to-freeze.txt"), "r") as f:
    REQUIRED = f.readlines()

with io.open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    LONG_DESCRIPTION = '\n' + f.read()

about = {}
with open(os.path.join(here, "glyph", "__version__.py")) as f:
    exec(f.read(), about)


class PublishCommand(Command):
    """Support setup.py publish."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds ...')
            rmtree(os.path.join(here, 'dist'))
        except FileNotFoundError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine...')
        os.system('twine upload dist/*')

        sys.exit()


if __name__ == '__main__':
    setup(
        name=NAME,
        version=about["__version__"],
        author=AUTHOR,
        author_email=EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENCE,
        keywords=KEYWORDS,
        url=URL,
        packages=find_packages(exclude=["tests", "doc", "examples"]),
        install_requires=REQUIRED,
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
        cmdclass={
        'publish': PublishCommand,
        },
    )
