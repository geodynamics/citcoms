#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



def temperature_diffadv(CitcomModule):
    from Advection_diffusion import Advection_diffusion
    return Advection_diffusion('temp','tsolver', CitcomModule)

    #from Temperature_diffadv import Temperature_diffadv
    #return Temperature_diffadv('temp','tsolver', CitcomModule)


# version
__id__ = "$Id: __init__.py,v 1.5 2003/07/25 20:43:29 tan2 Exp $"

# End of file
