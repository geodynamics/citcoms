#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



def fullSolver(name='full', facility='solver'):
    from FullSolver import FullSolver
    return FullSolver(name, facility)



def regionalSolver(name='regional', facility='solver'):
    from RegionalSolver import RegionalSolver
    return RegionalSolver(name, facility)



# version
__id__ = "$Id: __init__.py,v 1.1 2003/08/27 22:24:07 tan2 Exp $"

# End of file
