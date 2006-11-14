
from archimedes import use_merlin
use_merlin()

from merlin import setup, find_packages

setup(
    
    name = 'CitcomS', 
    version = '2.1',

    zip_safe = False,
    packages = find_packages(),
    
    install_requires = [
    'pythia[mpi] >= 0.8.1.0b4, < 0.8.2a',
    ],

    author = 'Louis Moresi, et al.',
    author_email = 'cig-mc@geodynamics.org',
    description = """A finite element mantle convection code.""",
    long_description = """CitcomS is a finite element code designed to solve thermal convection problems relevant to Earth's mantle. Written in C, the code runs on a variety of parallel processing computers, including shared and distributed memory platforms.""",
    license = 'GPL',
    url = 'http://www.geodynamics.org/cig/software/packages/mc/citcoms/',

)
