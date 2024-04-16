from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'subspyces - subspaces in python!'
LONG_DESCRIPTION = 'My first Python package with a slightly longer description'

# Setting up
setup(
    name="subspyces",
    version=VERSION,
    author="Matheus Lima",
    author_email="mlima@cvlab.cs.tsukuba.ac.jp",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    keywords=['python', 'subspyces'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
