#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



def fullSphere(CitcomModule):
    from FullSphere import FullSphere
    return FullSphere('full-sphere', 'mesher', CitcomModule)



def regionalSphere(CitcomModule):
    from RegionalSphere import RegionalSphere
    return RegionalSphere('regional-sphere', 'mesher', CitcomModule)



# version
__id__ = "$Id: __init__.py,v 1.3 2003/08/01 19:05:36 tan2 Exp $"

# End of file
