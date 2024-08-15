from setuptools import setup, find_packages

VERSION = '0.0.5'
DESCRIPTION = 'subspyces - subspaces in python!'
LONG_DESCRIPTION = ("Pytorch is a very powerfull framework for developing ML algorithms, "
                    "but implementing subspace methods in torch can be a bit repetitive "
                    "when we don't have a starting point. For this, we developed this simple "
                    "library which encapsulates some useful code that can be re-used and easily "
                    "integrated with other torch codebases.")

# Setting up
setup(
    name="subspyces",
    version=VERSION,
    author="Matheus Lima",
    author_email="mlima@cvlab.cs.tsukuba.ac.jp",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.8.0",
        "numpy>=1.22.2",
        "scikit_learn>=1.2.0",
        "setuptools>=68.2.2",
        "torch>=2.1.0",
        "torchvision>=0.16.0"
    ],
    keywords=['python', 'subspyces', 'pytorch'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Typing :: Typed"
    ]
)
