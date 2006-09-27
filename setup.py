
# This is not a normal 'setup.py' script; it is provided as a
# convenience to install Python packages required by CitcomS.  For
# instructions on installing CitcomS itself, see the file INSTALL.

try:
    # If setuptools 0.6b1 or later is installed, run with it.
    from pkg_resources import require
    require("setuptools>=0.6b1")
except:
    from ez_setup import use_setuptools
    use_setuptools()

import setuptools
import sys

requirements = []

if setuptools.bootstrap_install_from:
    requirements.append(setuptools.bootstrap_install_from)
    setuptools.bootstrap_install_from = None

requirements.append('pythia >= 0.8-1.0dev-r4617, < 0.8-2.0a, == dev')

setuptools.setup(
    script_args = (
    ['easy_install',
     '--find-links=svn://geodynamics.org/cig/cs/pythia/trunk#egg=pythia-dev'] +
    sys.argv[1:] +
    requirements
    )
)
