#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



def incompressibleNewtonian(CitcomModule):
    from Incompressible import Incompressible
    return Incompressible('incomp-newtonian', 'vsolver', CitcomModule)



def incompressibleNonNewtonian(CitcomModule):
    from Incompressible import Incompressible
    return Incompressible('incomp-non-newtonian', 'vsolver', CitcomModule)










# version
__id__ = "$Id: __init__.py,v 1.6 2003/08/22 22:14:59 tan2 Exp $"

# End of file
