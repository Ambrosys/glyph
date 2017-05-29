from setuptools import setup, find_packages

version = "0.3.2"


def read(fname, split=True):
    with open(fname, 'r') as f:
        content = f.read()
    return content.split('\n') if split else content


if __name__ == '__main__':
    setup(
        name='pyglyph',
        version=version,
        author='Markus Abel, Julien Gout, Markus Quade',
        author_email='markus.quade@ambrosys.de',
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
