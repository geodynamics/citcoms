
from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages

setup(
    
    name = 'CitcomS', 
    version = '2.1',

    zip_safe = False,
    packages = find_packages(),
    
    #setup_requires = [
    #'merlin',
    #],
    install_requires = [
    'pythia[mpi] >= 0.8.1.0b2, < 0.8.2a, == dev',
    ],
    extras_require = {
    'Exchanger': ['Exchanger >= 1.0, < 2.0a'],
    },

    author = 'Louis Moresi, et al.',
    author_email = 'cig-mc@geodynamics.org',
    description = """A finite element mantle convection code.""",
    long_description = """CitcomS is a finite element code designed to solve thermal convection problems relevant to Earth's mantle. Written in C, the code runs on a variety of parallel processing computers, including shared and distributed memory platforms.""",
    license = 'GPL',
    url = 'http://www.geodynamics.org/cig/software/packages/citcoms/',
    download_url = 'http://crust.geodynamics.org/~leif/shipping/', # temporary

)
