#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



def imcompressibleNewtonian(CitcomModule):
    from Imcompressible import Imcompressible
    return Imcompressible('imcomp-newtonian', 'vsolver', CitcomModule)



def imcompressibleNonNewtonian(CitcomModule):
    from Imcompressible import Imcompressible
    return Imcompressible('imcomp-non-newtonian', 'vsolver', CitcomModule)










# version
__id__ = "$Id: __init__.py,v 1.5 2003/08/15 18:47:24 tan2 Exp $"

# End of file
