#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



def incompressibleNewtonian(name="incomp-newtonian", facility='vsolver'):
    from Incompressible import Incompressible
    return Incompressible(name, facility)



def incompressibleNonNewtonian(name="incomp-non-newtonian", facility='vsolver'):
    from Incompressible import Incompressible
    return Incompressible(name, facility)










# version
__id__ = "$Id: __init__.py,v 1.8 2003/10/28 23:51:48 tan2 Exp $"

# End of file
