#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



def incompressibleNewtonian(name, facility='vsolver'):
    from Incompressible import Incompressible
    return Incompressible(name, facility)



def incompressibleNonNewtonian(name, facility='vsolver'):
    from Incompressible import Incompressible
    return Incompressible(name, facility)










# version
__id__ = "$Id: __init__.py,v 1.7 2003/08/27 20:52:47 tan2 Exp $"

# End of file
