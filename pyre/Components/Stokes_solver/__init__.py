#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



def imcompressibleNewtonian(CitcomModule):
    from Imcompressible import ImcompressibleNewtonian
    return ImcompressibleNewtonian('imcomp-newtonian', 'vsolver', CitcomModule)



def imcompressibleNonNewtonian(CitcomModule):
    from Imcompressible import ImcompressibleNonNewtonian
    return ImcompressibleNonNewtonian('imcomp-non-newtonian', 'vsolver', CitcomModule)










# version
__id__ = "$Id: __init__.py,v 1.4 2003/07/24 17:46:47 tan2 Exp $"

# End of file
