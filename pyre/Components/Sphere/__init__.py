#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



def fullSphere(name, facility='mesher'):
    from FullSphere import FullSphere
    return FullSphere(name, facility)



def regionalSphere(name, facility='mesher'):
    from RegionalSphere import RegionalSphere
    return RegionalSphere(name, facility)



# version
__id__ = "$Id: __init__.py,v 1.4 2003/08/27 20:52:47 tan2 Exp $"

# End of file
