#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



def fullSphere(name="full-sphere", facility='mesher'):
    from FullSphere import FullSphere
    return FullSphere(name, facility)



def regionalSphere(name="regional-sphere", facility='mesher'):
    from RegionalSphere import RegionalSphere
    return RegionalSphere(name, facility)



# version
__id__ = "$Id: __init__.py,v 1.5 2003/10/28 01:55:01 tan2 Exp $"

# End of file
